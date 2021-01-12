module.exports = {
    "env": {
        "browser": true,
        "es2021": true,
        "node": true
    },
    "globals": {
        "__webpack_public_path__": "writable",
        "sorttable": "readable"
    },
    "extends": [
        "eslint:recommended",
        "plugin:react/recommended"
    ],
    "settings": {
        "react": {
            "version": "detect"
        }
    },
    "parserOptions": {
        "ecmaFeatures": {
            "jsx": true
        },
        "ecmaVersion": 12,
        "sourceType": "module"
    },
    "plugins": [
        "react"
    ],
    "rules": {
    }
};
