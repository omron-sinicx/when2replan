import React from 'react';
import { render } from 'react-dom';

import { marked } from 'marked';
import markedKatex from 'marked-katex-extension';
marked.use(markedKatex({ throwOnError: false }));

export default class Results extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return this.props.results ? (
      <div className="uk-section">
        <h2 id="results">Results</h2>
        {this.props.results.map((result, idx) => {
          return (
            <div
              dangerouslySetInnerHTML={{
                __html: marked.parse(result.text),
              }}
            />
          );
        })}
      </div>
    ) : null;
  }
}
