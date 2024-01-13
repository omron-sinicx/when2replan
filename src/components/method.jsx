import React from 'react';
import { render } from 'react-dom';

class Content extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    if (this.props.title) return <h2>{this.props.title}</h2>;
    if (this.props.text) return <p>{this.props.text}</p>;
    if (this.props.image)
      return (
        <img
          src={require('../images/' + this.props.image)}
          className="uk-align-center uk-responsive-width"
          alt=""
        />
      );
    return null;
  }
}

export default class Method extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    return this.props.method ? (
      <div className="uk-section">
        {this.props.method.map((subsection, idx) => {
          return (
            <div key={'subsection-' + idx}>
              <Content title={subsection.title} />
              <Content image={subsection.image} />
              <Content text={subsection.text} />
            </div>
          );
        })}
      </div>
    ) : null;
  }
}
