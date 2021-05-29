import React from 'react';
import {render} from 'react-dom';

export default class Authors extends React.Component {
  constructor(props) {
      super(props);
  }

  render() {
    return (
      <div>
        <div className="uk-flex uk-flex-center uk-text-primary">
          {this.props.authors.map((author)=>{ return <span className="uk-margin-left"><a href={author.url}>{author.name}</a><sup>{author.affiliation.join(',')}</sup></span> })}
        </div>
        <div className="uk-text-meta uk-text-center">
          {this.props.affiliations.map((affiliation,idx)=>{ return <span className="uk-margin-left"><sup>{idx+1}</sup>{affiliation}</span>})}
        </div>
      </div>
    );
  }
}
