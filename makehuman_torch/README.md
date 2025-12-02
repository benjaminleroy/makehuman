# MakeHuman Torch Model

This directory contains code to work with a PyTorch implementation of a differentiable MakeHuman model.

## Running code that uses MakeHuman core

Whenever you run a set of code that leverages MakeHuman's core functionality, you need to run it from the `makehuman/makehuman` folder to correctly access makehuman modules.

## Running code with `python`

For this extension, you'll see a `pyproject.toml` file in the main directory. This file is used to manage dependencies and package information for Python projects. To install the required dependencies listed in the `pyproject.toml` file, you should use the `uv` package manager to leveage. 

**Pytorch Version**: Additionally, you'll notice that I originally wrote this code on a Intel Mac, so the torch dependencies where defined by a wheel (not provided), which also constrainted the `numpy` distribution.

## Usage

This code naturally falls under the MakeHuman license and usage terms (due to "copyleft" structure). Please refer to the main MakeHuman repository for more details on licensing and usage policies. *Note that it is this author's belief that this code itself does not rely on any external libraries used within MakeHuman that have conflicting licenses.*