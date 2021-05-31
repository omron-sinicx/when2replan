import React from 'react';
import {render} from 'react-dom';
import omronImg from '../images/omron.png';

class ContactCard extends React.Component {
  constructor(props) { super(props); }
  render() {
    return (
      <>
        <div className="uk-width-auto">
          <img className="uk-comment-avatar" src={this.props.author.icon} width="80" height="80" alt="" />
        </div>
        <div className="uk-width-expand">
          <h4 className="uk-comment-title uk-margin-remove"><a target="_blank" className="uk-link-reset" href={this.props.author.url}>{this.props.author.name}</a></h4>
          <ul className="uk-comment-meta uk-subnav uk-subnav-divider uk-margin-remove-top">
            <li><a href="#">{this.props.author.position}</a></li>
            <li><a href="https://www.omron.com/sinicx/" target="_blank">OMRON SINIC X</a></li>
          </ul>
        </div>
      </>
    );
  }
}

class OmronContactCard extends React.Component {
  constructor(props) { super(props); }
  render() {
    return (
      <>
        <div className="uk-width-auto uk-visible@m">
          <img className="uk-comment-avatar" src={omronImg} width="80" height="80" alt="" />
        </div>
        <div className="uk-width-expand uk-visible@m">
          <h4 className="uk-comment-title uk-margin-remove"><a className="uk-link-reset" href="#">contact@sinicx.com</a></h4>
          <ul className="uk-comment-meta uk-subnav uk-subnav-divider uk-margin-remove-top">
            <li><a href="https://www.omron.com/sinicx/" target="_blank">OMRON SINIC X</a></li>
          </ul>
        </div>
      </>
    );
  }
}

export default class Contact extends React.Component {
  constructor(props) { super(props); }

  render() {
    return (
      <div className="uk-section">
        <h2>Contact</h2>
        <div className="uk-grid-medium uk-flex-middle" data-uk-grid>
          {this.props.contact_ids.map((cid)=>{ return <ContactCard author={this.props.authors[cid]} /> })}
          <OmronContactCard />
        </div>
      </div>
    );
  }
}
