# AutoFill

This package provides ability to automatically fill-in data in tables using Programming by Example (PBE) techniques described in [FlashFill](https://www.microsoft.com/en-us/research/publication/automating-string-processing-spreadsheets-using-input-output-examples/). 


FlashFill is a program synthesis framework for generating string-processing programs from input-output examples. 
Given examples of some desired string transformation, FlashFill infers this transformation as a program which can applied to additional inputs for which the corresponding outputs are not available.
This package implements the core PBE algorithms described in FlashFill and integrates it with `DataFrames.jl`.

The primary use case for `AutoFill.jl` in its current form is for data manipulation, replacing writing tedious manipulation scripts and regex patterns with just providing examples of the required transformation.


## Usage

```julia
using AutoFill
using DataFrames

df = DataFrame([
    ("International Conference on Machine Learning", "ICML"),
    ("Robotics Science and Systems", "RSS"),
    ("International Conference on Software Engineering", missing)
])

# infers transformation and autofills missing values in column 2
filled_df, p = AutoFill.autofill(df, 2) 
```

The above snippet generates the following filled-in dataframe and the discovered program - 

```
Discovered program: AutoFill.Loop("w0", AutoFill.SubStr(1, AutoFill.Pos(AutoFill.Token[], AutoFill.Token[AutoFill.UpperTok], AutoFill.IntVar(0, 1, "w0", 5)), AutoFill.Pos(AutoFill.Token[], AutoFill.Token[AutoFill.UpperTok, AutoFill.NonUpperTok], AutoFill.IntVar(0, 1, "w0", 5))))

Filled in DataFrame:
3×2 DataFrame
 Row │ 1                                  2       
     │ String                             String? 
─────┼────────────────────────────────────────────
   1 │ International Conference on Mach…  ICML
   2 │ Robtics Science and Systems        RSS
   3 │ International Conference on Soft…  ICSE
```

A more interactive use case could be providing examples of a transformation for a larger set of data -

![](./assets/autofill_demo.gif)


## TODO

The package is still a work in progress and contributions are welcome. 
The current priorities are to implement the following -

- [ ] Better user-facing interfaces: Currently `AutoFill.autofill(::DataFrame, ::Int)`, is the only user interface for the package. 
    - [ ] `Tables.jl` interface
    - [ ] More granular functionalitiy such as providing examples directly, accessing intermediate stages of the synthesis algorithm (e.g. DAGs) etc...
    - [ ] Support for distributed application of discovered transforms for larger scale data
- [ ] Performance optimisations: The current implementation of the core algorithm can take signiificant amounts of time to discover transformations for longer examples. Various points where it it can be optimised are -
    - [ ] Pre-computing hashes of programs before unification
    - [ ] Adding handcrafted rules to limit applicability of certain regex token, hence reducing search size
    - [ ] Improving parts of the underlying algorithm
- [ ] Support for richer underlying expressions: 
    - [ ] Learning `Loops` containing `Concatenate` expressions. This is currently disabled as its inclusion in the framework drastically affects performance
    - [ ] Noise handling
    - [ ] Unsupervised transform learning (as described in [BlinkFill](https://www.microsoft.com/en-us/research/publication/blinkfill-semi-supervised-programming-by-example-for-syntactic-string-transformations/))

