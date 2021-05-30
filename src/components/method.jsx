import React from 'react';
import {render} from 'react-dom';
import methodImg from '../images/method.png';

export default class Method extends React.Component {
  constructor(props) { super(props); }

  render() {
    return (
      <div className="uk-section">
        <h2>Neural A*</h2>
        <img src={methodImg} className="uk-align-center uk-responsive-width" alt="" />
        <p>{this.props.method}</p>
      </div>
    );
  }
}
