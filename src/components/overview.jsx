import React from "react";
import { render } from "react-dom";
import teaserImg from "../images/teaser.png";

export default class Overview extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div className="uk-section">
        <img
          src={teaserImg}
          className="uk-align-center"
          width="700px"
          height=""
          alt=""
        />
        <h2>Overview</h2>
        <p>{this.props.overview}</p>
      </div>
    );
  }
}
