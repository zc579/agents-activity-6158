//! THE FILE YOUR AGENT MUST WRITE.
//!
//! This is the Rust translation of `reference/version.py`. The signatures
//! below are fixed — `src/main.rs` calls them and `evaluate.py` speaks to
//! `main.rs`. You may add anything you like *in addition* to these.

use std::cmp::Ordering;

/// A parsed semantic version.
///
/// `prerelease` and `build` hold the text *after* the `-` and `+`
/// respectively, with the separator stripped, or `None` when absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub prerelease: Option<String>,
    pub build: Option<String>,
}

/// Check if a character is alphanumeric or hyphen
fn is_identifier_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '-'
}

/// Check if a string is a valid numeric identifier (no leading zeros except "0")
fn is_valid_numeric_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    if s.chars().all(|c| c.is_ascii_digit()) {
        // Must be "0" or start with non-zero digit
        s == "0" || !s.starts_with('0')
    } else {
        false
    }
}

/// Check if a string is a valid prerelease identifier
fn is_valid_prerelease_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    
    // "0" or [1-9]\d* (numeric without leading zeros)
    if is_valid_numeric_identifier(s) {
        return true;
    }
    
    // \d*[a-zA-Z-][0-9a-zA-Z-]* (must contain at least one letter or hyphen)
    if s.chars().all(is_identifier_char) {
        s.chars().any(|c| c.is_ascii_alphabetic() || c == '-')
    } else {
        false
    }
}

/// Check if a string is a valid build identifier
fn is_valid_build_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    s.chars().all(is_identifier_char)
}

/// Parse a version string. Return `Err` with a short reason for anything
/// that is not a valid semver 2.0.0 string.
///
/// Reject, among others: leading zeroes (`01.0.0`), missing components
/// (`1.0`), negative numbers, empty identifiers (`1.0.0-`), and whitespace.
pub fn parse(s: &str) -> Result<Version, String> {
    if s.is_empty() {
        return Err("empty version string".to_string());
    }

    // Find the end of the version core (major.minor.patch)
    let mut pos = 0;
    let bytes = s.as_bytes();

    // Parse major version
    let major_end = pos;
    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        pos += 1;
    }
    if pos == major_end || pos == bytes.len() || bytes[pos] != b'.' {
        return Err(format!("{} is not valid SemVer string", s));
    }
    let major_str = &s[major_end..pos];
    let major = major_str.parse::<u64>()
        .map_err(|_| format!("{} is not valid SemVer string", s))?;
    
    // Check for leading zeros in major
    if major_str != "0" && major_str.starts_with('0') {
        return Err(format!("{} is not valid SemVer string", s));
    }

    pos += 1; // skip '.'

    // Parse minor version
    let minor_end = pos;
    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        pos += 1;
    }
    if pos == minor_end || pos == bytes.len() || bytes[pos] != b'.' {
        return Err(format!("{} is not valid SemVer string", s));
    }
    let minor_str = &s[minor_end..pos];
    let minor = minor_str.parse::<u64>()
        .map_err(|_| format!("{} is not valid SemVer string", s))?;
    
    // Check for leading zeros in minor
    if minor_str != "0" && minor_str.starts_with('0') {
        return Err(format!("{} is not valid SemVer string", s));
    }

    pos += 1; // skip '.'

    // Parse patch version
    let patch_end = pos;
    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        pos += 1;
    }
    if pos == patch_end {
        return Err(format!("{} is not valid SemVer string", s));
    }
    let patch_str = &s[patch_end..pos];
    let patch = patch_str.parse::<u64>()
        .map_err(|_| format!("{} is not valid SemVer string", s))?;
    
    // Check for leading zeros in patch
    if patch_str != "0" && patch_str.starts_with('0') {
        return Err(format!("{} is not valid SemVer string", s));
    }

    // Parse optional prerelease
    let prerelease = if pos < bytes.len() && bytes[pos] == b'-' {
        pos += 1;
        let pre_start = pos;
        let mut pre_end = pos;
        
        // Must have at least one identifier
        let mut found_identifier = false;
        
        while pre_end < bytes.len() && bytes[pre_end] != b'+' {
            pre_end += 1;
        }
        
        if pre_start == pre_end {
            return Err(format!("{} is not valid SemVer string", s));
        }
        
        let prerelease_str = &s[pre_start..pre_end];
        
        // Split by '.' and validate each identifier
        for identifier in prerelease_str.split('.') {
            if !is_valid_prerelease_identifier(identifier) {
                return Err(format!("{} is not valid SemVer string", s));
            }
            found_identifier = true;
        }
        
        if !found_identifier {
            return Err(format!("{} is not valid SemVer string", s));
        }
        
        pos = pre_end;
        Some(prerelease_str.to_string())
    } else {
        None
    };

    // Parse optional build
    let build = if pos < bytes.len() && bytes[pos] == b'+' {
        pos += 1;
        let build_start = pos;
        let build_end = bytes.len();
        
        if build_start == build_end {
            return Err(format!("{} is not valid SemVer string", s));
        }
        
        let build_str = &s[build_start..build_end];
        
        // Split by '.' and validate each identifier
        for identifier in build_str.split('.') {
            if !is_valid_build_identifier(identifier) {
                return Err(format!("{} is not valid SemVer string", s));
            }
        }
        
        pos = build_end;
        Some(build_str.to_string())
    } else {
        None
    };

    // Ensure we've consumed the entire string
    if pos != bytes.len() {
        return Err(format!("{} is not valid SemVer string", s));
    }

    Ok(Version {
        major,
        minor,
        patch,
        prerelease,
        build,
    })
}

/// Render a `Version` back to its canonical string form.
/// `parse(&to_string(&v))` must round-trip for every valid `v`.
pub fn to_string(v: &Version) -> String {
    let mut result = format!("{}.{}.{}", v.major, v.minor, v.patch);
    
    if let Some(ref pre) = v.prerelease {
        result.push('-');
        result.push_str(pre);
    }
    
    if let Some(ref build) = v.build {
        result.push('+');
        result.push_str(build);
    }
    
    result
}

/// Compare two prerelease version strings (dot-separated identifiers)
/// Returns: -1 if a < b, 0 if a == b, 1 if a > b
fn compare_prerelease(a: Option<&str>, b: Option<&str>) -> i32 {
    let a_parts: Vec<&str> = match a {
        Some(s) => s.split('.').collect(),
        None => vec![],
    };
    let b_parts: Vec<&str> = match b {
        Some(s) => s.split('.').collect(),
        None => vec![],
    };

    for (a_part, b_part) in a_parts.iter().zip(b_parts.iter()) {
        // Try to parse as numbers
        let a_num = a_part.parse::<u64>().ok();
        let b_num = b_part.parse::<u64>().ok();

        let cmp_result = match (a_num, b_num) {
            (Some(an), Some(bn)) => {
                // Both numeric: compare numerically
                if an < bn { -1 } else if an > bn { 1 } else { 0 }
            }
            (Some(_), None) => {
                // a is numeric, b is not: numeric is less
                -1
            }
            (None, Some(_)) => {
                // a is not numeric, b is: alphanumeric is greater
                1
            }
            (None, None) => {
                // Both alphanumeric: lexical comparison
                if a_part < b_part { -1 } else if a_part > b_part { 1 } else { 0 }
            }
        };

        if cmp_result != 0 {
            return cmp_result;
        }
    }

    // All shared identifiers are equal, compare lengths
    if a_parts.len() < b_parts.len() {
        -1
    } else if a_parts.len() > b_parts.len() {
        1
    } else {
        0
    }
}

/// Semver precedence.
///
/// Careful — this is where naive translations break:
///   * build metadata is IGNORED entirely for precedence
///   * a version WITH a prerelease is LOWER than the same version without
///   * prerelease identifiers compare left to right, dot-separated
///   * all-numeric identifiers compare numerically
///   * all other identifiers compare lexically in ASCII order
///   * numeric identifiers always rank LOWER than non-numeric ones
///   * if all preceding identifiers are equal, more identifiers wins
///
/// The spec's own worked example, which you should be able to reproduce:
///   1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-alpha.beta < 1.0.0-beta
///     < 1.0.0-beta.2 < 1.0.0-beta.11 < 1.0.0-rc.1 < 1.0.0
pub fn compare(a: &Version, b: &Version) -> Ordering {
    // Compare major, minor, patch
    if a.major != b.major {
        return if a.major < b.major {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    if a.minor != b.minor {
        return if a.minor < b.minor {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    if a.patch != b.patch {
        return if a.patch < b.patch {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }

    // Handle prerelease comparison
    // Key rule: a version WITHOUT prerelease is GREATER than one WITH prerelease
    match (&a.prerelease, &b.prerelease) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,  // No prerelease > with prerelease
        (Some(_), None) => Ordering::Less,     // With prerelease < no prerelease
        (Some(a_pre), Some(b_pre)) => {
            // Both have prerelease, compare them lexicographically
            let pre_cmp = compare_prerelease(Some(a_pre.as_str()), Some(b_pre.as_str()));
            if pre_cmp < 0 {
                Ordering::Less
            } else if pre_cmp > 0 {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
    }
}

/// Increment major; reset minor and patch; drop prerelease and build.
pub fn bump_major(v: &Version) -> Version {
    Version {
        major: v.major.saturating_add(1),
        minor: 0,
        patch: 0,
        prerelease: None,
        build: None,
    }
}

/// Increment minor; reset patch; drop prerelease and build.
pub fn bump_minor(v: &Version) -> Version {
    Version {
        major: v.major,
        minor: v.minor.saturating_add(1),
        patch: 0,
        prerelease: None,
        build: None,
    }
}

/// Increment patch; drop prerelease and build.
///
/// Do not guess the prerelease interaction — read `reference/version.py`
/// and check against the oracle. `evaluate.py` compares you to the real
/// `semver` package on every bump of every valid version it generates.
pub fn bump_patch(v: &Version) -> Version {
    Version {
        major: v.major,
        minor: v.minor,
        patch: v.patch.saturating_add(1),
        prerelease: None,
        build: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmp(a: &str, b: &str) -> Result<Ordering, String> {
        Ok(compare(&parse(a)?, &parse(b)?))
    }

    #[test]
    fn test_parse_simple() -> Result<(), String> {
        let v = parse("1.2.3")?;
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert_eq!(v.prerelease, None);
        assert_eq!(v.build, None);
        Ok(())
    }

    #[test]
    fn test_parse_with_prerelease() -> Result<(), String> {
        let v = parse("1.0.0-alpha")?;
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 0);
        assert_eq!(v.patch, 0);
        assert_eq!(v.prerelease, Some("alpha".to_string()));
        assert_eq!(v.build, None);
        Ok(())
    }

    #[test]
    fn test_parse_with_build() -> Result<(), String> {
        let v = parse("1.0.0+build")?;
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 0);
        assert_eq!(v.patch, 0);
        assert_eq!(v.prerelease, None);
        assert_eq!(v.build, Some("build".to_string()));
        Ok(())
    }

    #[test]
    fn test_parse_leading_zeros_rejected() -> Result<(), String> {
        assert!(parse("01.0.0").is_err());
        assert!(parse("1.02.0").is_err());
        assert!(parse("1.0.03").is_err());
        Ok(())
    }

    #[test]
    fn test_precedence_chain() -> Result<(), String> {
        assert_eq!(cmp("1.0.0-alpha", "1.0.0-alpha.1")?, Ordering::Less);
        assert_eq!(cmp("1.0.0-alpha.1", "1.0.0-alpha.beta")?, Ordering::Less);
        assert_eq!(cmp("1.0.0-alpha.beta", "1.0.0-beta")?, Ordering::Less);
        assert_eq!(cmp("1.0.0-beta", "1.0.0-beta.2")?, Ordering::Less);
        assert_eq!(cmp("1.0.0-beta.2", "1.0.0-beta.11")?, Ordering::Less);
        assert_eq!(cmp("1.0.0-beta.11", "1.0.0-rc.1")?, Ordering::Less);
        assert_eq!(cmp("1.0.0-rc.1", "1.0.0")?, Ordering::Less);
        Ok(())
    }

    #[test]
    fn test_build_ignored_in_compare() -> Result<(), String> {
        assert_eq!(cmp("1.0.0+a", "1.0.0+b")?, Ordering::Equal);
        assert_eq!(cmp("1.0.0", "1.0.0+build")?, Ordering::Equal);
        Ok(())
    }

    #[test]
    fn test_bump_major() -> Result<(), String> {
        let v = bump_major(&parse("1.2.3-rc.1+b")?);
        assert_eq!(v.major, 2);
        assert_eq!(v.minor, 0);
        assert_eq!(v.patch, 0);
        assert_eq!(v.prerelease, None);
        assert_eq!(v.build, None);
        Ok(())
    }

    #[test]
    fn test_bump_minor() -> Result<(), String> {
        let v = bump_minor(&parse("1.2.3-rc.1+b")?);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 3);
        assert_eq!(v.patch, 0);
        assert_eq!(v.prerelease, None);
        assert_eq!(v.build, None);
        Ok(())
    }

    #[test]
    fn test_bump_patch() -> Result<(), String> {
        let v = bump_patch(&parse("1.2.3-rc.1+b")?);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 4);
        assert_eq!(v.prerelease, None);
        assert_eq!(v.build, None);
        Ok(())
    }

    #[test]
    fn test_round_trip() -> Result<(), String> {
        let versions = vec![
            "0.0.0",
            "1.2.3",
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0-alpha.beta",
            "1.0.0+build",
            "1.0.0-rc.1+build.123",
        ];

        for version_str in versions {
            let v = parse(version_str)?;
            let s = to_string(&v);
            let v2 = parse(&s)?;
            assert_eq!(v, v2);
        }

        Ok(())
    }
}
