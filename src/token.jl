###################
# Language Tokens #
###################


# We use the following collection of character classes 𝐶: 
# - Numeric Digits (0-9), 
# - Alphabets (a-zA-Z), 
# - Lowercase alphabets (a-z), 
# - Uppercase alphabets (A-Z), 
# - Accented alphabets, 
# - Alphanumeric characters, 
# - Whitespace characters, 
# - All characters. 
# We use the following SpecialTokens:
# - StartTok: Matches the beginning of a string. 
# - EndTok: Matches the end of a string.
# - A token for each special character, 
#   such as hyphen, dot, semicolon, colon, 
#   comma, backslash, forwardslash, 
#   left/right parenthesis/bracket etc.


# For better readability, we reference tokens by representative names. 
# For example, AlphTok refers to a sequence of alphabetic characters,
# NumTok refers to a sequence of numeric digits, NonDigitTok refers to 
# a sequence of characters that are not numeric digits, HyphenTok matches 
# with the hyphen character.

@enum Token begin
    AlphNumTok
    AlphNumWsTok
    AlphTok
    BackSlashTok
    ColonTok
    CommaTok
    DashTok
    DotTok
    EndTok
    HyphenTok
    LeftAngleTok
    LeftParenTok
    LeftSquareTok
    UnderscoreTok
    LowerTok
    NonAlphNumTok
    NonAlphNumWsTok
    NonAlphTok
    NonLowerTok
    NonNumTok
    NonUpperTok
    NumTok
    RightAngleTok
    RightParenTok
    RightSquareTok
    SemicolonTok
    SlashTok
    StartTok
    UpperTok
    WsTok
end

const TokenSet = Set{Token}
const TokenSeq = Vector{Token}

const token_to_r_string = Dict(
    AlphNumTok => "[a-zA-Z0-9]+",
    AlphNumWsTok => "[a-zA-Z0-9 ]+",
    AlphTok => "[a-zA-Z]+",
    BackSlashTok => "\\\\",
    ColonTok => ":",
    CommaTok => ",",
    DashTok => "-",
    DotTok => "\\.",
    EndTok => "\$",
    HyphenTok => "-",
    LeftAngleTok => "<",
    LeftParenTok => "\\(",
    LeftSquareTok => "<",
    UnderscoreTok => "_",
    LowerTok => "[a-z]+",
    NonAlphNumTok => "[^a-zA-Z0-9]+",
    NonAlphNumWsTok => "[^a-zA-Z0-9 ]+",
    NonAlphTok => "[^a-zA-Z]+",
    NonLowerTok => "[^a-z]+",
    NonNumTok => "[^\\d]+",
    NonUpperTok => "[^A-Z]+",
    NumTok => "\\d+",
    RightAngleTok => ">",
    RightParenTok => "\\)",
    RightSquareTok => ">",
    SemicolonTok => ";",
    SlashTok => "\\/",
    StartTok => "^",
    UpperTok => "[A-Z]+",
    WsTok => " ",
)

tokseq_to_r_string(tokseq) = join([
    token_to_r_string[tok] for tok in tokseq
])

const r_string_to_token = Dict(
    r_str => tok for (tok, r_str) in pairs(token_to_r_string) 
)

const token_to_r = Dict(
    tok => Regex(r_str) for (tok, r_str) in pairs(token_to_r_string) 
)


# # Uncomment below to pretty-print regex
# Base.show(io::IO, tok::Token) = print(io, token_to_r_string[tok])
# Base.show(io::IO, tokseq::TokenSeq) = print(
#     io,
#     join([token_to_r_string[tok] for tok in tokseq])
# )
