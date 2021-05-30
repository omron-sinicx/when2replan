import React from 'react';
import {render} from 'react-dom';
import teaserImg from '../images/teaser.png';

export default class Overview extends React.Component {
  constructor(props) { super(props); }

  render() {
    return (
      <div className="uk-section">
        <div className="uk-flex uk-flex-center">
          <img src={teaserImg} width="400" height="" alt="" />
        </div>
        <h2>Overview</h2>
        <p>{this.props.overview}</p>
      </div>
    );
  }
}
