import React from "react";
import PropTypes from "prop-types";
import {Route, NavLink, Redirect} from "react-router-dom";
import {CatchLink} from "../../../components/CatchLink";

import ContentPage from "../content/installation";

const installMethods = {
    "installation-homebrew": {
        method: "Homebrew"
    },
    "installation-docker": {
        method: "Docker"
    },
    "installation-source": {
        method: "Source",
    },
    "installation-source-windows": {
        method: "Source",
    },
    "installation-yum": {
        method: "yum/dnf",
    },
    "installation-apt-ubuntu": {
        method: "apt (Ubuntu)",
    },
    "installation-apt-debian": {
        method: "apt (Debian)",
    },
    "installation-windows": {
        method: "Installer (.zip)"
    }

};

const introTexts = {
    mac: <span>In addition to the following install methods for MacOS, Timescale also offers Docker images prebuilt with <CatchLink to="/getting-started/installation/mac/installation-docker#postgis-docker">PostGIS</CatchLink> or our <CatchLink to="/getting-started/installation/mac/installation-docker#prometheus-docker">Prometheus adapter</CatchLink>.</span>,
    linux: <span>In addition to the following install methods for Linux, Timescale also offers Docker images prebuilt with <CatchLink to="/getting-started/installation/linux/installation-docker#postgis-docker">PostGIS</CatchLink> or our <CatchLink to="/getting-started/installation/linux/installation-docker#prometheus-docker">Prometheus adapter</CatchLink>.</span>,
    windows: <span>In addition to the following install methods for Windows, Timescale also offers Docker images prebuilt with <CatchLink to="/getting-started/installation/windows/installation-docker#postgis-docker">PostGIS</CatchLink> or our <CatchLink to="/getting-started/installation/windows/installation-docker#prometheus-docker">Prometheus adapter</CatchLink>.</span>
};

const DefaultContent = () => (<div></div>);

class OSChooser extends React.Component {
    constructor(props) {
        super(props);
        this.state = {InstallContent: DefaultContent, importPath: ""};
    }

    componentDidUpdate(preProps, preState) {
        if (this.state.importPath + this.props.location.hash !== preState.importPath + preProps.location.hash) {
            this.props.scrollToAnchor();
        }
    }

    render() {
        const {match, page} = this.props;
        return (
            <div>
                <Route exact path={match.path} render={() => (
                    <Redirect to={`${match.url}/${page.children[0].href}/${page.children[0].children[0].href}`}/>
                )}/>
                <Route exact path={match.path + "/:os"} render={(props) => {
                    let childPage = page.children.find((lilPage) => (lilPage.href === props.match.params.os));
                    return <Redirect to={`${match.url}/${props.match.params.os}/${childPage.children[0].href}`}/>;
                }}/>

                <h1>Installation</h1>
                <div className="test-chooser">
                    <div className="test-chooser__os-menu">
                        <NavLink
                        className={"os-button"}
                        to={`/${match.params.version}/getting-started/installation/mac`}>
                            <img src="//assets.iobeam.com/images/docs/Apple_logo_black.svg" height="50"/>
                            <div className="test-chooser__os-menu-label">Mac</div>
                        </NavLink>
                        <NavLink
                        className={"os-button"}
                        to={`/${match.params.version}/getting-started/installation/linux`}>
                            <img src="//assets.iobeam.com/images/docs/Tux.svg" height="50"/>
                            <div className="test-chooser__os-menu-label">Linux</div>
                        </NavLink>
                        <NavLink
                        className={"os-button"}
                        to={`/${match.params.version}/getting-started/installation/windows`}>
                            <img src="//assets.iobeam.com/images/docs/Windows_logo_-_2012.svg" height="50"/>
                            <div className="test-chooser__os-menu-label">Windows</div>
                        </NavLink>
                    </div>
                    <Route path={match.path + "/:os"} render={(props) => {
                        let OSPage = page.children.filter((childPage) => {
                            return childPage.href === props.match.params.os;
                        })[0];

                        const installLinks = OSPage.children.map((installPage) => {
                            return <InstallLink key={installPage.href} page={installPage} url={props.match.url}/>;
                        });
                        return (
                            <div className="adventure ">
                                <div className="os hidden__index">Installation for {props.match.params.os}</div>
                                <p>{introTexts[props.match.params.os]}</p>
                                <div className="adventure__button-list">
                                    {installLinks}
                                </div>
                                    <Route path={`${props.match.path}/:method`} render={(propers) => {

                                        let importPath = `../content/${propers.match.params.method}`;
                                        if (importPath !== this.state.importPath) {
                                            import(`../content/${propers.match.params.method}`)
                                            .then((InstallComponent) => {
                                                console.log("import install subpage");
                                                this.setState({InstallContent: InstallComponent.default, importPath});
                                            })
                                            .catch(() => {
                                                console.warn("failed to load install content");
                                            });
                                        }

                                        return (
                                            <div className="adventure__content">
                                                <div className="method hidden__index">Installation {props.match.params.method}</div>
                                                <this.state.InstallContent/>
                                            </div>
                                        );
                                    }}/>
                            </div>
                        );
                    }}/>


                </div>
                <ContentPage/>
            </div>
        );
    }
}

OSChooser.propTypes = {
    location: PropTypes.object.isRequired,
    match: PropTypes.object,
    page: PropTypes.object,
    scrollToAnchor: PropTypes.func
};

const InstallLink = ({page, url}) => {
    return (
        <NavLink
        to={`${url}/${page.href}`}
        className="adventure__button">
            <div className="adventure__button-text">
                {installMethods[page.href].method}
            </div>
        </NavLink>
    );
};

InstallLink.propTypes = {
    page: PropTypes.object,
    url: PropTypes.string
};

export default OSChooser;
