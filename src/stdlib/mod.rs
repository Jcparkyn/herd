pub fn load_stdlib_module(name: &str) -> Option<&'static str> {
    match name {
        "@system" => Some(include_str!("system.herd")),
        "@basics" => Some(include_str!("basics.herd")),
        "@bitwise" => Some(include_str!("bitwise.herd")),
        "@dict" => Some(include_str!("dict.herd")),
        "@io" => Some(include_str!("io.herd")),
        "@list" => Some(include_str!("list.herd")),
        "@math" => Some(include_str!("math.herd")),
        "@random" => Some(include_str!("random.herd")),
        "@regex" => Some(include_str!("regex.herd")),
        "@string" => Some(include_str!("string.herd")),
        "@file" => Some(include_str!("file.herd")),
        "@parallel" => Some(include_str!("parallel.herd")),
        _ => None,
    }
}
