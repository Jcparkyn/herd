pub fn parse_string_literal(s: &str) -> String {
    s[1..s.len() - 1].to_string()
}
