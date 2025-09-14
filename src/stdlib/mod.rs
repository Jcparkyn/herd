pub fn load_stdlib_module(name: &str) -> &'static str {
    match name {
        "@system" => include_str!("system.herd"),
        "@basics" => include_str!("basics.herd"),
        "@bitwise" => include_str!("bitwise.herd"),
        "@dict" => include_str!("dict.herd"),
        "@io" => include_str!("io.herd"),
        "@list" => include_str!("list.herd"),
        "@math" => include_str!("math.herd"),
        "@random" => include_str!("random.herd"),
        "@regex" => include_str!("regex.herd"),
        "@string" => include_str!("string.herd"),
        _ => panic!("No such stdlib module: {}", name),
    }
}
