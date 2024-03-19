import React from 'react';
import { render } from 'react-dom';
import { FaAddressCard, FaEnvelope, FaGithub } from 'react-icons/fa6';

class ContactCard extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <div className="uk-width-1-2@s uk-flex">
        <div className="uk-width-auto uk-margin-right">
          <a target="_blank" href={this.props.author.url}>
          <FaAddressCard size="3em" color="#1C5EB8" />
          </a>
        </div>
        <div className="uk-width-expand">
          <h4 className="uk-comment-title uk-margin-remove">
            <a
              target="_blank"
              className="uk-link-reset"
              href={this.props.author.url}
            >
              {this.props.author.name}
            </a>
          </h4>
          <ul className="uk-comment-meta uk-subnav uk-subnav-divider uk-margin-remove-top">
            <li className="uk-visible@m">
              <a href="#">{this.props.author.position}</a>
            </li>
            <li>
              <a href="https://www.omron.com/sinicx/" target="_blank">
                OMRON SINIC X
              </a>
            </li>
          </ul>
        </div>
      </div>
    );
  }
}

class OmronContactCard extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <div className="uk-width-1-2@s uk-flex">
        <div className="uk-width-auto uk-margin-right">
          <FaEnvelope size="3em" color="#1C5EB8" />
        </div>
        <div className="uk-width-expand">
          <h4 className="uk-comment-title uk-margin-remove">
            <a className="uk-link-reset" >
              contact@sinicx.com
            </a>
          </h4>
          <ul className="uk-comment-meta uk-subnav uk-subnav-divider uk-margin-remove-top">
            <li>
              <a href="https://www.omron.com/sinicx/" target="_blank">
                OMRON SINIC X
              </a>
            </li>
          </ul>
        </div>
      </div>
    );
  }
}

class GithubContactCard extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <div className="uk-width-1-2@s uk-flex">
        <div className="uk-width-auto uk-margin-right">
          <a target="_blank" href={this.props.url}>
          <FaGithub size="3em" color="#1C5EB8" />
          </a>
        </div>
        <div className="uk-width-expand">
          <h4 className="uk-comment-title uk-margin-remove">
            <a className="uk-link-reset" target="_blank" href={this.props.url}>
              GitHub issues
            </a>
          </h4>
        </div>
      </div>
    );
  }
}
export default class Contact extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div className="uk-section">
        <h2>Contact</h2>
        <div className="uk-grid-medium" data-uk-grid>
          {this.props.contact_ids.map((cid) => {
            if (cid == 'omron') {
              return <OmronContactCard />
            } else if (cid == 'github'){
              return <GithubContactCard url={this.props.resources.code+'/issues'}/>
            } else {
              return (
                <ContactCard
                  author={this.props.authors[cid-1]}
                  key={'contact-' + cid-1}
                />
              );
            }
          })}
        </div>
      </div>
    );
  }
}
