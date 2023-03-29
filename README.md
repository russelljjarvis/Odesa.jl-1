<h1 align="center">
  Odesa.jl: Julia implementation of Feast/fully connected ODESA
</h1>




<p align="center">
  <a href="#Getting-Started">Getting Started</a> •
  <a href="#Entry-Points">Entry Points</a> •
  <a href="#Performance-Issues">Performance Issues</a> •
  <a href="#Design-TODO">Design TODO</a> •

  
</p>


<!---
For this to work (direct to build status of this repository fork), you would need to fiddle around with manually setting up actions.

![Build status](https://github.com/yeshwanthravitheja/julia_odesa/actions/workflows/ci.yml/badge.svg](https://github.com/yeshwanthravitheja/julia_odesa/actions/workflows/ci.yml/badge.svg)
--->
![https://github.com/russelljjarvis/Odesa.jl/actions/workflows/ci.yml/badge.svg](https://github.com/russelljjarvis/Odesa.jl/actions/workflows/ci.yml/badge.svg)


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



### Getting Started

<details>
  <summary>Install the Julia module</summary>
    
This is not yet an official package, so the package would need to be added in developer mode. The short way to do this is as follows:
```
import Pkg
Pkg.add(url="https://github.com/russelljjarvis/Odesa.jl.git")
```
or Original:
```  
Pkg.add(url="https://github.com/yeshwanthravitheja/julia_odesa.git")
 ```
or 
```
  
] add https://github.com/russelljjarvis/Odesa.jl
```
The long way invovles:
```
git clone https://github.com/russelljjarvis/Odesa.jl
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
