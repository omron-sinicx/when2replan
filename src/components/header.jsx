import React from 'react';
import {render} from 'react-dom';

export default class Header extends React.Component {
  constructor(props) { super(props); }

  render() {
    return (
      <div className="uk-cover-container uk-background-secondary">
        <div className="uk-container uk-container-small uk-section">
          <div className="uk-text-center uk-text-bold">
            <p className="uk-h1 uk-text-primary">{this.props.title}</p>
            <span className="uk-label uk-label-primary uk-text-center uk-margin-bottom">{this.props.conference}</span>
          </div>
          <div className="uk-flex uk-flex-center uk-text-primary">
            <span className="">Ryo Yonetani*<sup>1</sup></span>
            <span className="uk-margin-left">Tatsunori Taniai*<sup>1</sup></span>
            <span className="uk-margin-left">Mohammadamin Barekatain<sup>1,2</sup></span>
            <span className="uk-margin-left">Mai Nishimura<sup>1</sup></span>
            <span className="uk-margin-left">Asako Kanezaki<sup>3</sup></span>
          </div>
          <div className="uk-text-meta uk-text-center" uk-grid>
            <span className="uk-width-1-4"><sup>1</sup>OMRON SINIC X</span>
            <span className="uk-width-1-4"><sup>2</sup>Technical University of Munich</span>
            <span className="uk-width-1-4"><sup>3</sup>Tokyo Institute of Technology</span>
            <span className="uk-width-1-4">*Equal Contribution.</span>
          </div>
          <div className="uk-flex uk-flex-center uk-margin-top">
            <button className="uk-button uk-button-text">
              <span className="uk-icon" uk-icon="file-pdf" /> paper
            </button>
            <button className="uk-button uk-button-text uk-margin-left">
              <span className="uk-icon" uk-icon="play-circle" /> video
            </button>
            <button className="uk-button uk-button-text uk-margin-left" uk-tooltip="title: COMING SOON">
              <span className="uk-icon" uk-icon="github" /> code
            </button>
          </div>
        </div>
      </div>
    );
  }
}
