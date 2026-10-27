"""AWS Braket devices through the Amazon Braket PennyLane plugin.

Requires ``pip install -e ".[braket]"`` and AWS credentials resolved by boto3
(``AWS_PROFILE`` / ``AWS_REGION`` or the default credential chain).
Device ARNs are listed in the Braket console, for example the managed
simulator ``arn:aws:braket:::device/quantum-simulator/amazon/sv1``.
"""

import pennylane as qml

from qfd.hardware.errors import HardwareConfigurationError


def braket_device(
    device_arn: str,
    n_qubits: int,
    shots: int,
    s3_bucket: str | None = None,
    s3_prefix: str = "qfd",
) -> qml.devices.Device:
    """Create a Braket QPU or managed-simulator device.

    Broadcast batches are submitted as parallel quantum tasks.

    Args:
        device_arn: Braket device ARN.
        n_qubits: Number of wires.
        shots: Shots per task; QPUs require a positive value.
        s3_bucket: Results bucket; ``None`` uses the Braket default bucket.
        s3_prefix: Key prefix for results.

    Returns:
        The PennyLane device.

    Raises:
        HardwareConfigurationError: If the plugin is not installed or shots
            are not positive.
    """
    if shots <= 0:
        raise HardwareConfigurationError(
            "Braket devices need a positive number of shots"
        )
    kwargs: dict[str, object] = {
        "device_arn": device_arn,
        "wires": n_qubits,
        "shots": shots,
        "parallel": True,
    }
    if s3_bucket is not None:
        kwargs["s3_destination_folder"] = (s3_bucket, s3_prefix)
    try:
        return qml.device("braket.aws.qubit", **kwargs)
    except qml.exceptions.DeviceError as exc:
        raise HardwareConfigurationError(
            'Install the Braket extra: pip install -e ".[braket]"'
        ) from exc
