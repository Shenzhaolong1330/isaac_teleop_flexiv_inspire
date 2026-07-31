from pathlib import Path
import yaml

from flexiv_inspire_isaac.system_config import load_system_config, render_runtime_configs
from flexiv_inspire_xr_bridge.streamer import EncoderSettings, build_ffmpeg_command


ROOT = Path(__file__).resolve().parents[1]


def test_ffmpeg_rtp_contract_matches_isaac_receiver():
    settings = EncoderSettings("/usr/bin/ffmpeg", "127.0.0.1", 5000, 30.0, 4.0, 15)
    for encoder in ("h264_nvenc", "libx264"):
        command = build_ffmpeg_command(settings, encoder)
        assert command[command.index("-payload_type") + 1] == "96"
        assert command[-1] == "rtp://127.0.0.1:5000?pkt_size=1200"
        assert "-bf" in command and command[command.index("-bf") + 1] == "0"


def test_rendered_bridge_and_receiver_use_identical_ports_and_rates(tmp_path):
    config = load_system_config(ROOT / "config" / "site.yaml")
    rendered = render_runtime_configs(config, tmp_path)
    bridge = yaml.safe_load(rendered["xr_bridge.yaml"].read_text())["flexiv_inspire_xr_bridge"]["ros__parameters"]
    receiver = yaml.safe_load(rendered["isaac_camera_receiver.yaml"].read_text())
    assert receiver["source"] == "rtp"
    for name, camera in config.document["cameras"]["streams"].items():
        assert bridge[f"streams.{name}.fps"] == float(camera["fps"])
        assert bridge[f"streams.{name}.port"] == receiver["cameras"][name]["streams"]["mono"]["port"]
        assert receiver["cameras"][name]["width"] == camera["width"]
        assert receiver["cameras"][name]["height"] == camera["height"]
