{ pkgs, lib, ... }:

let
  # Linux uses nixpkgs' Zig, which links against and runs on the Nix store's
  # glibc. On macOS, nixpkgs' zig_0_17 dies with SIGBUS, so it is the official
  # ziglang.org build: the same tarball and SHA-256 as
  # .github/scripts/install-zig.sh.
  zigVersion = "0.17.0";
  zigMacos = {
    triple = "aarch64-macos";
    sha256 = "b607e9b9234790a008116ae5bdb71c6243b84b9fb42a53a9e70fde41c06c536a";
  };
  zig =
    if pkgs.stdenv.hostPlatform.system == "aarch64-darwin" then
      pkgs.stdenvNoCC.mkDerivation {
        pname = "zig";
        version = zigVersion;
        src = pkgs.fetchurl {
          url = "https://ziglang.org/download/${zigVersion}/zig-${zigMacos.triple}-${zigVersion}.tar.xz";
          inherit (zigMacos) sha256;
        };
        dontConfigure = true;
        dontBuild = true;
        dontFixup = true;
        installPhase = ''
          mkdir -p $out/bin $out/share/zig
          cp -r . $out/share/zig
          ln -s $out/share/zig/zig $out/bin/zig
        '';
      }
    else
      pkgs.zig_0_17;
  # The DSPy reference harness (tools/dspy_reference) needs CPython 3.11; uv
  # installs DSPy 3.5.0 from uv.lock, which pins every package by sha256.
  python = pkgs.python311;
in
{
  packages = [
    zig
    python
    pkgs.uv
  ]
  ++ lib.optionals pkgs.stdenv.hostPlatform.isDarwin [ pkgs.xcbuild ];

  env = {
    UV_PYTHON = "${python}/bin/python3";
    UV_PYTHON_DOWNLOADS = "never";
  }
  # NumPy's manylinux wheel (pulled in by Optuna, which MIPROv2 uses) loads
  # libstdc++ and libz from the system; Nix shells have no system copies.
  // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
    ];
  };

  scripts = {
    dz-test.exec = ''zig build test --summary all "$@"'';
    dz-examples.exec = ''zig build examples --summary all "$@"'';
    dz-dspy.exec = ''tools/dspy_reference/check.sh'';
    # Re-record real-model transcripts (not run by CI). Credentials come from the
    # SecretSpec `live` profile (Keychain), never from committed files:
    # (tools/live, not the root: devenv auto-loads a root secretspec.toml):
    #   secretspec --file tools/live/secretspec.toml set OPENAI_API_KEY --profile live
    dz-live.exec = ''
      set -euo pipefail
      if [ -z "''${DZ_LIVE_SECRETS:-}" ]; then
        exec env DZ_LIVE_SECRETS=1 secretspec --file "$DEVENV_ROOT/tools/live/secretspec.toml" run \
          --profile live --reason "dszig live inference recording" -- dz-live "$@"
      fi
      (cd tools/dspy_reference && uv run --locked live.py \
        ../../tests/fixtures/live/dspy_3.5.0.json "openai/''${DSZIG_LIVE_MODEL:-gpt-4o-mini}")
      zig build replay -- tests/fixtures/live/dspy_3.5.0.json
      zig build live -- tests/fixtures/live/dszig_3.5.0.json
      (cd tools/dspy_reference && uv run --locked check_live.py ../../tests/fixtures/live/dszig_3.5.0.json)
      tools/dspy_reference/check.sh
    '';
    # One real CoT request (2 + 2) through the native LM, via SecretSpec.
    dz-live-smoke.exec = ''
      exec secretspec --file "$DEVENV_ROOT/tools/live/secretspec.toml" run \
        --profile live --reason "dszig live inference smoke" -- zig build live -- --smoke "$@"
    '';
    dz-check.exec = ''
      set -euo pipefail
      zig fmt --check .
      for mode in Debug ReleaseSafe; do
        dz-test -Doptimize="$mode"
        dz-examples -Doptimize="$mode"
      done
      dz-dspy
    '';
  };

  # Zig skips SDK detection when NIX_CFLAGS_COMPILE or NIX_LDFLAGS is set.
  enterShell =
    lib.optionalString pkgs.stdenv.hostPlatform.isDarwin ''
      unset NIX_CFLAGS_COMPILE NIX_CFLAGS_LINK NIX_LDFLAGS
    ''
    + ''
      echo "dszig: zig $(zig version), $(python3 --version), uv $(uv --version | cut -d' ' -f2)"
      echo "  dz-test, dz-examples, dz-dspy (cross-check with DSPy 3.5.0), dz-check (what CI runs), dz-live / dz-live-smoke (real model via SecretSpec)"
    '';

  enterTest = ''
    set -euo pipefail
    test "$(zig version)" = "${zigVersion}"
    grep -q ${zigMacos.sha256} .github/scripts/install-zig.sh
    python3 -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
    dz-check
  '';
}
