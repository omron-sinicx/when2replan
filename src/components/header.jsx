import React from 'react';
import {render} from 'react-dom';
import Authors from '../components/authors.jsx';

import {FontAwesomeIcon} from '@fortawesome/react-fontawesome';
import {faGithub,faYoutube,faMedium} from '@fortawesome/free-brands-svg-icons';
import {faFilePdf} from '@fortawesome/free-solid-svg-icons';

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
            <a className="uk-button uk-button-text" href={this.props.paper} target="_blank">
              <FontAwesomeIcon icon={faFilePdf} size="lg" color="#1C5EB8" />
              <span className="uk-margin-small-left uk-margin-small-right uk-text-primary uk-text-bolder">paper</span>
            </a>
            <a className="uk-button uk-button-text uk-margin-medium-left" href={this.props.code} target="_blank">
              <FontAwesomeIcon icon={faGithub} size="lg" color="#1C5EB8" />
              <span className="uk-margin-small-left uk-margin-small-right uk-text-primary uk-text-bolder">code</span>
            </a>
            <a className="uk-button uk-button-text uk-margin-medium-left" href="#video">
              <FontAwesomeIcon icon={faYoutube} size="lg" color="#1C5EB8" />
              <span className="uk-margin-small-left uk-margin-small-right uk-text-primary uk-text-bolder">video</span>
            </a>
            <a className="uk-button uk-button-text uk-margin-medium-left" href={this.props.blog}>
              <FontAwesomeIcon icon={faMedium} size="lg" color="#1C5EB8" />
              <span className="uk-margin-small-left uk-margin-small-right uk-text-primary uk-text-bolder">blog</span>
            </a>
          </div>
        </div>
      </div>
    );
  }
}
