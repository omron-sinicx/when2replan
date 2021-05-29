import React from 'react';
import {render} from 'react-dom';
import {Helmet} from "react-helmet";

import Header from "../components/header.jsx";
import Overview from "../components/overview.jsx";
import Video from "../components/video.jsx";
import Method from "../components/method.jsx";

import data from "../data/template.yaml";

console.log(data);

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
            { property: "og:image:width", content: "912"},
            { property: "og:image:height", content: "618"},
            { property: 'og:url', content: data.url},
            { name: 'twitter:card', content: "summary_large_image"},
            { name: 'twitter:title', content: data.title},
            { name: 'twitter:image', content: data.image},
            { name: 'twitter:description', content: data.description},
            { name: 'twitter:url', content: data.url},
            { name: 'twitter:site', content: data.twitter},
          ]}
        />
        <Header title={data.title}
                conference={data.conference}
                authors={data.authors}
                affiliations={data.affiliations}
                code={data.repo}
                video={data.video}
                paper={data.paper} />
        <div className="uk-container uk-container-small">
          <Overview overview={data.overview} />
          <Video video={data.video} />
          <Method method={data.method} />
        </div>
      </div>
    )
  }
}

render(<Template/>, document.getElementById('root'))
