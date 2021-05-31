## osx-project-page-template
- A project page template based on [UIKit](https://getuikit.com/) + [React](https://ja.reactjs.org/)

### Setup
- Install latest `node.js` and `npm`

##### Ubuntu / WSL2 Ubuntu

```sh
$ sudo apt update
$ sudo apt install -y nodejs npm
$ nodejs -v
v10.19.0
$ npm -v
7.5.1
```

##### Mac OS X

```sh
$ brew install nodebrew
$ nodebrew install-binary latest
$ echo 'export PATH=$PATH:$HOME/.nodebrew/current/bin' >> ~/.bash_profile
$ source ~/.bash_profile
```

### Build
- open `localhost:8080` by your browser

```sh
$ npm install # install dependencies
$ npm run build
$ npm run serve
```

### Develop

```sh
$ npm run serve
```

### Customize
- modify UIKit variables at `src/scss/theme.scss`
  - see https://github.com/uikit/uikit/blob/bc6dd1851652e5b77387a1efefc16cea6e3d165b/src/scss/variables.scss

### Structure

```
src/
├── components          # React components loaded in index.jsx
│   ├── authors.jsx
│   ├── citation.jsx
│   ├── contact.jsx
│   ├── footer.jsx
│   ├── header.jsx
│   ├── method.jsx
│   ├── overview.jsx
│   ├── results.jsx
│   └── video.jsx
├── data
│   └── template.yaml    # template arguments
├── html
│   └── index.html
├── images               # images to be relocated to assets/ by file-loader
│   ├── method.png
│   ├── result1.png
│   ├── result2.png
│   └── teaser.png
├── js
│   └── styles.js        # embed styles to js
├── pages
│   └── index.jsx        # template root
└── scss                 # color theme zoo
    ├── dark-theme.scss
    └── theme.scss
```

### Template
- fillin values at `src/data/template.yaml`
- fillin `null` for N/A contents (e.g. `method: null`)

```yaml
organization: OMRON SINIC X
twitter: "@omron_sinicx"
title: Path Planning using Neural A* Search
conference: ICML2021
resources:
  paper: https://arxiv.org/abs/1909.13111
  code: https://github.com/omron-sinicx/multipolar
  video: https://www.youtube.com/embed/adUnIj83RtU
  blog: https://medium.com/sinicx/multipolar-multi-source-policy-aggregation-for-transfer-reinforcement-learning-between-diverse-bc42a152b0f5
description: a Japanese stop-motion short anime series produced by Shin-Ei Animation and Japan Green Hearts in cooperation with Bandai Namco Entertainment.
image: https://omron-sinicx.github.io/neural-astar/assets/teaser.png
url: https://omron-sinicx.github.io/neural-astar
speakerdeck: b7a0614c24014dcbbb121fbb9ed234cd   # speakerdeck embed id
authors:
  - name: Ryo Yonetani*
    affiliation: [1]
    position: principal investigator
    url: https://yonetaniryo.github.io/
  - name: Tatsunori Taniai*
    affiliation: [1]
    position: senior researcher
  - name: Mohammadamin Barekatain
    affiliation: [1, 2]
    url: http://barekatain.me/
  - name: Mai Nishimura
    affiliation: [1]
    url: https://denkiwakame.github.io
  - name: Asako Kanezaki
    affiliation: [2]
    url: https://kanezaki.github.io/
contact_ids: [1]   # 0=1st author 1=2nd author
affiliations:
  - OMRON SINIC X
  - Technical University of Munich
  - Tokyo Institute of Technology
meta:
  - "* work done as an intern at OMRON SINIC X."
bibtex: >
  @article{DBLP:journals/corr/abs-2009-07476,
    author    = {Ryo Yonetani and
                 Tatsunori Taniai and
                 Mohammadamin Barekatain and
                 Mai Nishimura and
                 Asako Kanezaki},
    title     = {Path Planning using Neural A* Search},
    journal   = {CoRR},
    volume    = {abs/2009.07476},
    year      = {2020},
    url       = {https://arxiv.org/abs/2009.07476},
    archivePrefix = {arXiv},
    eprint    = {2009.07476},
    timestamp = {Wed, 23 Sep 2020 15:51:46 +0200},
    biburl    = {https://dblp.org/rec/journals/corr/abs-2009-07476.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
  }
```

### Available Style Components in UIKit
- see https://getuikit.com/docs/introduction

### GitHub Actions
- generate personal access token
  - see https://github.com/peaceiris/actions-gh-pages#%EF%B8%8F-set-personal-access-token-personal_token
- register the token as `PERSONAL_TOKEN` at `https://github.com/path/to/your/repo/settings/secrets/actions`
- `$ git remote add github {your-github-repo-path}`
- `$ git push github main:project-page`
- set `Source` `Branch=gh-pages` `/(root)` at `https://github.com/path/to/your/repo/settings/pages`
