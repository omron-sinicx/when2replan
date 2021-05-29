import React from 'react';
import {render} from 'react-dom';
import {Helmet} from "react-helmet";
import Header from "../components/header.jsx";
import ForkMeOnGitHub from "../components/fork-me-on-github.jsx"

import data from "../data/template.json";

class Template extends React.Component {
  render() {
    return (
      <div>
        <Helmet
          title={data.title}
          meta={[
            { property: "og:site_name", content: data.organization },
            { property: "og:type", content: "article" },
            { property: "og:title", content: data.title },
            { property: "og:description", content: data.description},
            { property: 'og:image', content: data.image}, // FIXME
            { name: 'twitter:card', content: "summary_large_image"},
            { name: 'twitter:title', content: data.title},
            { name: 'twitter:image', content: data.image},
            { name: 'twitter:description', content: data.description},
            { name: 'twitter:url', content: data.url},
            { name: 'twitter:site', content: data.twitter},
          ]}
        />
        <Header title={data.title} conference={data.conference} />
      </div>
    )
  }
}

render(<Template/>, document.getElementById('root'))
