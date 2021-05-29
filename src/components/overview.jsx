import React from 'react';
import {render} from 'react-dom';

export default class Overview extends React.Component {
  constructor(props) { super(props); }

  render() {
    return (
      <div className="uk-section">
        <div className="uk-margin-top uk-flex uk-flex-center">
          <img data-src="assets/teaser.png" width="400" height="" alt="" />
        </div>
        <h2>Overview</h2>
        <p>{this.props.overview}</p>
      </div>
    );
  }
}
