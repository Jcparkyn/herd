pub fn load_stdlib_module(name: &str) -> &'static str {
    match name {
        "@basics" => include_str!("basics.bovine"),
        "@bitwise" => include_str!("bitwise.bovine"),
        "@dict" => include_str!("dict.bovine"),
        "@io" => include_str!("io.bovine"),
        "@list" => include_str!("list.bovine"),
        "@math" => include_str!("math.bovine"),
        _ => panic!("No such stdlib module: {}", name),
    }
}
