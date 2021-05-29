import React from 'react';
import {render} from 'react-dom';

class ContactCard extends React.Component {
  constructor(props) { super(props); }
  render() {
    return (
      <div uk-grid="true">
        <div className="uk-width-auto">
          <img className="uk-comment-avatar" src={this.props.author.icon} width="80" height="80" alt="" />
        </div>
        <div className="uk-width-expand">
          <h4 className="uk-comment-title uk-margin-remove"><a target="_blank" className="uk-link-reset" href={this.props.author.url}>{this.props.author.name}</a></h4>
          <ul className="uk-comment-meta uk-subnav uk-subnav-divider uk-margin-remove-top">
            <li><a href="#">{this.props.author.position}</a></li>
            <li><a href="https://www.omron.com/sinicx/">OMRON SINIC X</a></li>
          </ul>
        </div>
      </div>
    );
  }
}

export default class Contact extends React.Component {
  constructor(props) { super(props); }

  render() {
    return (
      <div className="uk-section">
        <h2>Contact</h2>
        <ContactCard author={this.props.authors[0]} />
      </div>
    );
  }
}
