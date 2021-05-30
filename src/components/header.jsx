import React from 'react';
import {render} from 'react-dom';
import Authors from '../components/authors.jsx';

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
              <span className="uk-icon" uk-icon="file-pdf" /> paper
            </a>
            <a className="uk-button uk-button-text uk-margin-medium-left" href="#video">
              <span className="uk-icon" uk-icon="play-circle" /> video
            </a>
            <a className="uk-button uk-button-text uk-margin-medium-left" uk-tooltip="title: COMING SOON" href={this.props.code} target="_blank">
              <span className="uk-icon" uk-icon="github" /> code
            </a>
          </div>
        </div>
      </div>
    );
  }
}
