## osx-project-page-template
- A project page template based on [UIKit](https://getuikit.com/)

### Setup
- Install latest `node.js` and `npm`

#### Ubuntu / WSL2 Ubuntu

```sh
$ sudo apt update
$ sudo apt install -y nodejs npm
$ nodejs -v
v10.19.0
$ npm -v
7.5.1
```

#### Mac OS X

```sh
$ brew install nodebrew
$ nodebrew install-binary latest
$ echo 'export PATH=$PATH:$HOME/.nodebrew/current/bin' >> ~/.bash_profile
```

### Build
- localhost `localhost:8080`

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

### Available Components
- see https://getuikit.com/docs/introduction
