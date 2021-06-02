import React from "react";
import { render } from "react-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faAddressCard, faEnvelope } from "@fortawesome/free-solid-svg-icons";

class ContactCard extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <div class="uk-width-1-2@s uk-flex">
        <div className="uk-width-auto uk-margin-right">
          <FontAwesomeIcon icon={faAddressCard} size="3x" color="#1C5EB8" />
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
      <div class="uk-width-1-2@s uk-flex">
        <div className="uk-width-auto uk-margin-right">
          <FontAwesomeIcon icon={faEnvelope} size="3x" color="#1C5EB8" />
        </div>
        <div className="uk-width-expand">
          <h4 className="uk-comment-title uk-margin-remove">
            <a className="uk-link-reset" href="#">
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
            return <ContactCard author={this.props.authors[cid]} />;
          })}
          <OmronContactCard />
        </div>
      </div>
    );
  }
}
