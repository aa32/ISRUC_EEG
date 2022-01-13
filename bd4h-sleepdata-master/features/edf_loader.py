import numpy as np
import os
import re
import pyedflib
from urllib.request import urlretrieve, urlopen


class PhysiobankEDFLoader(object):

    def __init__(self):
        self.url = 'https://physionet.org/physiobank/database/sleep-edfx/'

    @property
    def psg_record_paths(self):
        if not os.path.exists('./data/RECORDS'):
            urlretrieve(os.path.join(self.url, 'RECORDS'), './data/RECORDS')
        with open('data/RECORDS') as f:
            records = f.readlines()
        records = [record.strip() for record in records]
        return records

    def load_sc_records(self, save=False):
        '''
        Use RECORDS file to load all the sleep-cassette PSGs. Because that file doesn't
        contain the hypnogram file names I have to parse the index html to look it up >_>
        Set save flag to download all the files, it takes a while.

        return: path to edf, eg. data/sleep-cassette/SC4001E0-PSG.edf
        '''
        print("Loading records ...")
        records = []
        html = urlopen(os.path.join(self.url, 'sleep-cassette')).read().decode('utf-8')
        psg_paths = [path for path in self.psg_record_paths if 'sleep-cassette' in path]

        if not os.path.exists('./data/sleep-cassette/'):
            os.mkdir('./data/sleep-cassette/')

        for psg_path in psg_paths:
            # PSG
            local_psg_path = os.path.join('data/', psg_path)
            if not os.path.exists(local_psg_path) and save:
                edf_url = os.path.join(self.url, psg_path)
                urlretrieve(edf_url, local_psg_path)

            # Hypnogram
            pattern = os.path.split(psg_path)[1].replace('0-PSG', '.-Hypnogram')
            hyp_filename = re.search(re.compile(pattern), html).group(0)
            if not hyp_filename:
                raise Exception('Cannot find hypnogram file for %s' % psg_path)
            local_hyp_path = os.path.join('data/sleep-cassette', hyp_filename)
            if not os.path.exists(local_hyp_path) and save:
                edf_url = os.path.join(self.url, 'sleep-cassette', hyp_filename)
                urlretrieve(edf_url, local_hyp_path)

            records.append((local_psg_path, local_hyp_path))
        return records

    def print_record(self, edf_path):
        reader = pyedflib.EdfReader(edf_path)

        print("\n======= %s =======\n" % edf_path)
        print("edfsignals: %i" % reader.signals_in_file)
        print("file duration: %i seconds" % reader.file_duration)
        print("startdate: %i-%i-%i" % (reader.getStartdatetime().day,reader.getStartdatetime().month,reader.getStartdatetime().year))
        print("starttime: %i:%02i:%02i" % (reader.getStartdatetime().hour,reader.getStartdatetime().minute,reader.getStartdatetime().second))
        print("patientcode: %s" % reader.getPatientCode())
        print("gender: %s" % reader.getGender())
        print("birthdate: %s" % reader.getBirthdate())
        print("patient_name: %s" % reader.getPatientName())
        print("patient_additional: %s" % reader.getPatientAdditional())
        print("admincode: %s" % reader.getAdmincode())
        print("technician: %s" % reader.getTechnician())
        print("equipment: %s" % reader.getEquipment())
        print("recording_additional: %s" % reader.getRecordingAdditional())
        print("datarecord duration: %f seconds" % reader.getFileDuration())
        print("number of datarecords in the file: %i" % reader.datarecords_in_file)
        print("number of annotations in the file: %i\n" % reader.annotations_in_file)

        annotations = reader.readAnnotations()
        for n in np.arange(reader.annotations_in_file):
            print("annotation: onset is %f    duration is %s    description is %s" % (annotations[0][n],annotations[1][n],annotations[2][n]))

        for channel in range(reader.signals_in_file):
            print("signal parameters for the %d.channel:\n" % channel)
            print("label: %s" % reader.getLabel(channel))
            print("samples in file: %i" % reader.getNSamples()[channel])
            print("physical maximum: %f" % reader.getPhysicalMaximum(channel))
            print("physical minimum: %f" % reader.getPhysicalMinimum(channel))
            print("digital maximum: %i" % reader.getDigitalMaximum(channel))
            print("digital minimum: %i" % reader.getDigitalMinimum(channel))
            print("physical dimension: %s" % reader.getPhysicalDimension(channel))
            print("prefilter: %s" % reader.getPrefilter(channel))
            print("transducer: %s" % reader.getTransducer(channel))
            print("samplefrequency: %f\n" % reader.getSampleFrequency(channel))

            buf = reader.readSignal(channel)
            n = 200
            print("read %i samples\n" % n)
            result = ""
            for i in np.arange(n):
                result += ("%.1f, " % buf[i])
            print(result)
            print("\n")
