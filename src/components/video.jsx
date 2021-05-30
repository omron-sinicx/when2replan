import React from 'react';
import {render} from 'react-dom';

export default class Video extends React.Component {
  constructor(props) { super(props); }

  render() {
    return (
      <div className="uk-section">
        <h2>Video</h2>
        <iframe className="uk-align-center uk-width-2xlarge" width="600" height="300" src={this.props.video} frameBorder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen></iframe>
      </div>
    );
  }
}
