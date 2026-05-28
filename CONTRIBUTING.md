# Contributing to Scaluq

Thank you for your interest in Scaluq!

There are many ways to contribute, not just by writing code.  
This document provides a overview of how to contribute.

## Ask Questions
If you have questions about Scaluq, feel free to ask via Issues, and the community will assist you!  
Your question will become valuable knowledge for others seeking for help.  
The issue title should be prefixed with `[Question]` so that community can easily identify it.  

## Report Bugs
Reporting problems are always welcome! Open an Issue and let us know what went wrong.
The issue title should be prefixed with `[Bug]`.  
Please include the following details if possible:
- Steps to reproduce the issue
- Expected behavior and actual behavior
- A code snippet demonstrating the issue
- Scaluq version, OS information, etc.

## Request Features
Have a new feature in mind? Open an Issue to let us know!  
The issue title should be prefixed with `[Feat]`.


## Pull Requests

It is highly recommended to use the provided devcontainer for development.

### Pull Request Process
Pull Requests (PRs) are always welcome!  
Opening an issue first is not strictly necessary (though recommended) for a single, self-contained PR. You can simply submit a PR directly to the `main` branch.  
However, if you plan to make substantial changes that require multiple PRs, please discuss your proposed changes in an Issue first.

### Size of Pull Request
When you create a PR, the manually written diff (excluding automatically generated files) should ideally not exceed about 300 lines.  
If you need to submit a larger PR, please consult the maintainers first so a dedicated feature branch can be created for review.

## Requirements for Merging a PR

- The patch is legally compliant under the project's LICENSE.
- All GitHub Actions checks pass.
- At least 1 maintainer approves the PR.

### Pull Request Title Examples  
- new feature: `Add function-name`
- update: `Update ci-version`
- bug: `Fix bug-name`


## How to test

We strongly recommend testing your code locally before submitting PR.  
First you have to install [dependencies](https://github.com/qulacs/scaluq) to your computer
before building and testing your patch.  
Then, execute the following commands in the root directory of the project:  

### For devcontainer

```console
$ export SCALUQ_USE_TEST=ON 
$ ./script/configure.sh
$ ninja -C build
$ ./build/tests/scaluq_test
```
