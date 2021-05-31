import React from 'react';
import {render} from 'react-dom';
import Authors from '../components/authors.jsx';

import {FontAwesomeIcon} from '@fortawesome/react-fontawesome';
import {faGithub,faYoutube,faMedium} from '@fortawesome/free-brands-svg-icons';
import {faFilePdf} from '@fortawesome/free-solid-svg-icons';

class ResourceBtn extends React.Component {
  constructor(props) {
    super(props);
    this.icons = {
      "paper": faFilePdf,
      "code": faGithub,
      "video": faYoutube,
      "blog": faMedium,
    }
  }
  render() {
    if (!this.props.url) return null;
    const aClass = this.props.title == "paper" ? `uk-button uk-button-text`: `uk-button uk-button-text uk-margin-medium-left`; // FIXME
    return (
      <>
        <a className={aClass} href={this.props.url} target="_blank">
          <FontAwesomeIcon icon={this.icons[this.props.title]} size="lg" color="#1C5EB8" />
          <span className="uk-margin-small-left uk-margin-small-right uk-text-primary uk-text-bolder">{this.props.title}</span>
        </a>
      </>
    );
  }
}

export default class Header extends React.Component {
  constructor(props) { super(props); }

  render() {
    const titleClass = `uk-${this.props.title.length > 100 ? 'h2' : 'h1'} uk-text-primary`;
    return (
      <div className="uk-cover-container uk-background-secondary">
        <div className="uk-container uk-container-small uk-section">
          <div className="uk-text-center uk-text-bold">
            <p className={titleClass}>{this.props.title}</p>
            <span className="uk-label uk-label-primary uk-text-center uk-margin-bottom">{this.props.conference}</span>
          </div>
          <Authors authors={this.props.authors} affiliations={this.props.affiliations} meta={this.props.meta}/>
          <div className="uk-flex uk-flex-center uk-margin-top">
            {Object.keys(this.props.resources).map(key => <ResourceBtn url={this.props.resources[key]} title={key} />)}
          </div>
        </div>
      </div>
    );
  }
}
