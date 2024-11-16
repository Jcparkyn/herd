pub fn load_stdlib_module(name: &str) -> &'static str {
    match name {
        "@basics" => include_str!("basics.herd"),
        "@bitwise" => include_str!("bitwise.herd"),
        "@dict" => include_str!("dict.herd"),
        "@io" => include_str!("io.herd"),
        "@list" => include_str!("list.herd"),
        "@math" => include_str!("math.herd"),
        "@random" => include_str!("random.herd"),
        _ => panic!("No such stdlib module: {}", name),
    }
}
