<p align="center">
  <img width = "250" src="docs/VPLanetLogo.png"/>
</p>

<h1 align="center">MaxLEV: Maximum Likelihood Estimation for VPLanet</h1>

<p align="center">
  <a href="https://RoryBarnes.github.io/MaxLEV/"><img src="https://img.shields.io/badge/read-the_docs-blue.svg?style=flat"></a>
  <a href="https://github.com/RoryBarnes/MaxLEV/actions/workflows/docs.yml">
  <img src="https://github.com/RoryBarnes/MaxLEV/actions/workflows/docs.yml/badge.svg">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-purple.svg"></a>
    <a href="https://VirtualPlanetaryLaboratory.github.io/vplanet/conduct.html"><img src="https://img.shields.io/badge/Code%20of-Conduct-7d93c7.svg"></a><br>
  <a href="https://github.com/RoryBarnes/MaxLEV/actions/workflows/tests.yml">
  <img src="https://github.com/RoryBarnes/MaxLEV/actions/workflows/tests.yml/badge.svg">
  <a href="https://codecov.io/gh/RoryBarnes/MaxLEV">
  <img src="https://codecov.io/gh/RoryBarnes/MaxLEV/branch/main/graph/badge.svg">
  </a>
    <img src="https://img.shields.io/badge/Python-3.9--3.12-orange.svg"></a><br>
  <a href="https://github.com/RoryBarnes/MaxLEV/actions/workflows/pip-install.yml">
  <img src="https://github.com/RoryBarnes/MaxLEV/actions/workflows/pip-install.yml/badge.svg">
    <img src = "https://img.shields.io/badge/Platforms-Linux_|%20macOS-darkgreen.svg?style=flat">
  </a>
</p>

`MaxLEV` finds the maximum likelihood estimates of [`VPLanet`](https://github.com/VirtualPlanetaryLaboratory/vplanet) model parameters given observational constraints. With `MaxLEV` you can quickly identify the best-fit initial conditions for a planetary system by comparing simulated outputs to observed values with asymmetric uncertainties. After finding the maximum likelihood solution, use [`alabi`](https://github.com/dflemin3/alabi) to obtain posterior distributions, and [`vconverge`](https://github.com/VirtualPlanetaryLaboratory/vconverge) to derive posteriors for derived quantities. [Read the docs](https://RoryBarnes.github.io/MaxLEV/) to learn how to perform maximum likelihood estimation with VPLanet.
