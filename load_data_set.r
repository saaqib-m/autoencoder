usethis::create_github_token(host = "https://github.ic.ac.uk")
tkn <- "" # The token you generated
devtools::install_github("tanaka-group/TanakaData", host = "github.ic.ac.uk/api/v3", auth_token = tkn) 
