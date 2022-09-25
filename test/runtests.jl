using AutoFill
using DataFrames
using Test

@testset "AutoFill" begin
    test_dfs = [
        (DataFrame([
            ("h(2)kb3c", "2"),
            ("c2d(4)sk", "4"),
            ("sd3f(8)a6sd3f", missing)
        ]), "8"),

        (DataFrame([
            ("BTR KRNL WK CORN 15Z", "15Z"),
            ("CAMP DRY DBL NDL 36 OZ", "36 OZ"),
            ("CHORE BOY HD SC SPNG 1 PK", "1 PK"),
            ("FRENCH WORCESTERSHIRE 5 Z", "5 Z"),
            ("O F TOMATO PASTE 6 OZ", missing),
        ]), "6 OZ"),

        (DataFrame([
            ("International Business Machines", "IBM"),
            ("British Broadcasting Corporation", "BBC"),
            ("National Public Radio", missing)
        ]), "NPR"),

        (DataFrame([
            ("International Conference on Machine Learning", "ICML"),
            ("Robtics Science and Systems", "RSS"),
            ("International Conference on Software Engineering", missing)
        ]), "ICSE"),
    ]

    for (test_df, ans) in test_dfs
        println("Data with missing:")
        println(test_df)
        filled_df, p = AutoFill.autofill(test_df, 2)
        @test !isnothing(filled_df) && !isnothing(ans) && ans == filled_df[size(filled_df)[1], 2]
        if isnothing(p)
            println("No programs found!") 
        else
            println("Program found: ", p)
            println(filled_df)
        end
        println()
    end 
end