using FlashFill
using DataFrames
using Test

@testset "FlashFill.jl" begin

    test_dfs = [
        DataFrame([
            ("h(2)kb3c", "2"),
            ("c2d(4)sk", "4"),
            ("sd3f(8)a6sd3f", missing)
        ]),

        DataFrame([
            ("BTR KRNL WK CORN 15Z", "15Z"),
            ("CAMP DRY DBL NDL 3.6 OZ", "3.6 OZ"),
            ("CHORE BOY HD SC SPNG 1 PK", "1 PK"),
            ("FRENCH WORCESTERSHIRE 5 Z", "5 Z"),
            ("O F TOMATO PASTE 6 OZ", missing),
        ]),

        DataFrame([
            ("Company\\Code\\index.html", "Company\\Code\\"),
            ("Company\\Docs\\Spec\\specs.doc", "Company\\Docs\\Spec\\"),
            ("Company\\Users\\person.doc", missing)
        ]),

        DataFrame([
            ("International Business Machines", "IBM"),
            ("British Broadcasting Corporation", "BBC"),
            ("National Public Radio", missing)
        ])
    ]

    for test_df in test_dfs
        println("Data with missing:")
        println(test_df)
        filled_df, p = FlashFill.flashfill(test_df, 2)
        if isnothing(p)
            println("No programs found!") 
        else
            println("Program found: ", p)

            println(filled_df)
        end
        println()
    end 

end
