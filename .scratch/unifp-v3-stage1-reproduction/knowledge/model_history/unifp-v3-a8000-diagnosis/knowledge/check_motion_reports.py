"""Return a nonzero status when real motion evidence misses confirmed limits."""

import argparse
import json
from pathlib import Path


def main() -> None:
    """Check recorded motion errors and base contacts against D04 limits.

    Raises:
        OSError: A requested native-physics report cannot be read.
        KeyError: A report lacks required motion evidence.
        SystemExit: A measured error or base contact exceeds the confirmed limit.

    Note:
        Prints machine-readable per-case verdicts; never changes the reports or
        simulator. Runtime success alone does not satisfy this physical gate.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', type=Path, nargs='+')
    arguments = parser.parse_args()
    verdicts = []
    for report_path in arguments.reports:
        report = json.loads(report_path.read_text())
        for case_name, axis in (('forward', 0), ('backward', 0), ('lateral', 1), ('yaw', 2)):
            case = report['cases'][case_name]
            tolerance = 0.05 if axis == 2 else 0.03
            measured_error = case['body_velocity_error_mean'][axis]
            verdicts.append({
                'report': report_path.name, 'case': case_name,
                'primary_mae': measured_error, 'tolerance': tolerance,
                'peak_base_contact_N': case['peak_base_contact_N'],
                'passed': measured_error <= tolerance and case['peak_base_contact_N'] <= 1.0,
            })
    passed = all(case['passed'] for case in verdicts)
    print(json.dumps({'passed': passed, 'cases': verdicts}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
