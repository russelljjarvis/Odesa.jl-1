<h1 align="center">
  Odesa.jl: Julia implementation of Feast/fully connected ODESA
</h1>


<p align="center">
  <a href="#Getting-Started">Getting Started</a> •
  <a href="#Entry-Points">Entry Points</a> •
  <a href="#Performance-Issues">Performance Issues</a> •
  <a href="#Design-TODO">Design TODO</a> •

  
</p>

[![CI](https://github.com/russelljjarvis/Odesa.jl-1/actions/workflows/ci.yml/badge.svg)](https://github.com/russelljjarvis/Odesa.jl-1/actions/workflows/ci.yml)






<!---
For this to work (direct to build status of this repository fork), you would need to fiddle around with manually setting up actions.



--->


To Install

```
] add "https://github.com/russelljjarvis/Odesa.jl-1"
```

Or

```
using Pkg
Pkg.add(url="https://github.com/russelljjarvis/Odesa.jl-1")
```
![image](https://user-images.githubusercontent.com/7786645/228419246-be765377-5d9e-424a-ae5a-1ffe2722eae0.png)

###

Works fine with 16Bit Floats (see image below).

![image](https://user-images.githubusercontent.com/7786645/228737546-f2547327-feed-43e8-ad3e-8d000cfd1b71.png)


### Getting Started

<details>
  <summary>Install the Julia module</summary>
    
This is not yet an official package, so the package would need to be added in developer mode. The short way to do this is as follows:
```
import Pkg
Pkg.add(url="https://github.com/russelljjarvis/Odesa.jl-1.git")
```
or Original:
```  
Pkg.add(url="https://github.com/yeshwanthravitheja/Odesa.jl-1.git")
 ```
or 
```
  
] add https://github.com/russelljjarvis/Odesa.jl-1
```
The long way invovles:
```
git clone https://github.com/russelljjarvis/Odesa.jl-1
```

```
cd Odesa.jl


#### To install Odesa permanently in development mode:

julia
]
(@v1.5) pkg> develop .
```
Or
```
Pkg.develop(PackageSpec(path=pwd()))

```

#### To install Odesa only for one session:

```
julia
import Pkg;Pkg.activate(".")
```

</details>

<details>
<summary>Entry Points</summary>


Experimental build of Odesa in julia
