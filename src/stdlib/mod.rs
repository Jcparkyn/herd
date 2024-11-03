pub fn load_stdlib_module(name: &str) -> &'static str {
    match name {
        "@list" => include_str!("list.bovine"),
        _ => panic!("No such stdlib module: {}", name),
    }
}
