#!/usr/bin/env python

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirdesign.html
import numpy
import scipy.signal
import pylab
import argparse
from sys import exit

parser = argparse.ArgumentParser(
    description = 'Calculate IIR filter coefficients.')

parser.add_argument(
    '-s', '--samplerate',
    type=float,
    help='sample rate',
    default=88200.)
parser.add_argument(
    '-c', '--cutoff-hz',
    type=float,
    help='Cutoff frequency (Hz)',
    default=22050.)
parser.add_argument(
    '-t', '--transition-width-hz',
    type=float,
    help='Transition width from passband to stopband (Hz)',
    default=2000.)
parser.add_argument(
    '-p', '--passband-ripple-db',
    type=float,
    help='Maximum passband ripple (dB)',
    default=1.)
parser.add_argument(
    '-a', '--att-db',
    type=float,
    help='Stopband attenuation dB',
    default=60.)
parser.add_argument(
    '--filter-type',
    type=str,
    choices=['butter', 'cheby1', 'cheby2', 'ellip', 'bessel'],
    help='IIR filter type',
    default='butter')
parser.add_argument(
    '--forced-order',
    type=int,
    help='Force order on the filter. When set, uses iirfilter instead of iirdesign',
    default=None)
parser.add_argument(
    '--plot',
    help='Plot graphs',
    action='store_true',
    default=False)
parser.add_argument(
    '--sos',
    help='Output as second-order sections (recommended for stability)',
    action='store_true',
    default=False)
parser.add_argument(
    '--float',
    help='Print the coefficients as float.',
    action='store_true',
    default=False)

args = parser.parse_args()

samplerate = args.samplerate
cutoff_hz = args.cutoff_hz
att_db = args.att_db
passband_ripple_db = args.passband_ripple_db
transwidth_hz = args.transition_width_hz
nyquist = samplerate * 0.5

# Normalized frequencies for iirdesign (0 to 1, where 1 is Nyquist)
wp = cutoff_hz / nyquist  # passband edge
ws = (cutoff_hz + transwidth_hz) / nyquist  # stopband edge

# Clamp to valid range
wp = min(wp, 0.99)
ws = min(ws, 0.99)

if args.forced_order:
    # Use iirfilter with forced order
    if args.sos:
        sos = scipy.signal.iirfilter(
            args.forced_order,
            cutoff_hz,
            btype='lowpass',
            ftype=args.filter_type,
            fs=samplerate,
            output='sos')
        order = args.forced_order
    else:
        b, a = scipy.signal.iirfilter(
            args.forced_order,
            cutoff_hz,
            btype='lowpass',
            ftype=args.filter_type,
            fs=samplerate,
            output='ba')
        order = args.forced_order
else:
    # Use iirdesign for automatic order calculation
    if args.sos:
        sos = scipy.signal.iirdesign(
            wp, ws,
            passband_ripple_db, att_db,
            ftype=args.filter_type,
            output='sos')
        order = sos.shape[0] * 2  # Each SOS section is 2nd order
    else:
        b, a = scipy.signal.iirdesign(
            wp, ws,
            passband_ripple_db, att_db,
            ftype=args.filter_type,
            output='ba')
        order = len(a) - 1

varname = f'iir_{args.filter_type}_'
varname += f'sr{int(samplerate)}hz_'
varname += f'fc{int(cutoff_hz)}hz_'
varname += f'tw{int(transwidth_hz)}hz_'
varname += f'att{int(att_db)}db'

vartype = 'float' if args.float else 'double'
coeff_substr = 'f' if args.float else ''

if args.sos:
    # Output as second-order sections
    num_sections = sos.shape[0]
    print(f'// Second-order sections: {num_sections} sections, 6 coefficients each')
    print(f'// Format per section: [b0, b1, b2, a0, a1, a2]')
    print(f'static std::array<std::array<{vartype}, 6>, {num_sections}> {varname}_sos = {{{{')
    for i, section in enumerate(sos):
        section_str = ', '.join(
            f'{c:.17g}{coeff_substr}' if c != 0. else f'0.{coeff_substr}'
            for c in section
        )
        comma = ',' if i < num_sections - 1 else ''
        print(f'  {{{{{section_str}}}}}{comma}')
    print(f'}}}};')
else:
    # Output as transfer function coefficients (b, a)
    print(f'// Numerator coefficients (b)')
    print(f'static std::array<{vartype}, {len(b)}> {varname}_b = {{')
    for coeff in b:
        if coeff == 0.:
            print(f'  0.{coeff_substr},')
        else:
            print(f'  {coeff:.17g}{coeff_substr},')
    print(f'}};')

    print(f'// Denominator coefficients (a)')
    print(f'static std::array<{vartype}, {len(a)}> {varname}_a = {{')
    for coeff in a:
        if coeff == 0.:
            print(f'  0.{coeff_substr},')
        else:
            print(f'  {coeff:.17g}{coeff_substr},')
    print(f'}};')

print(f'')
print(f'Filter type: {args.filter_type}')
print(f'Cutoff (Hz): {cutoff_hz}')
print(f'Transition width (Hz): {transwidth_hz}')
print(f'Passband ripple (dB): {passband_ripple_db}')
print(f'Stopband attenuation (dB): {att_db}')
print(f'Order: {order}')

if not args.plot:
    exit(0)

#------------------------------------------------
# Plot the magnitude response of the filter.
#------------------------------------------------
pylab.figure(1)
pylab.clf()
pylab.title('Frequency Response')
pylab.xlabel('Frequency (Hz)')
pylab.ylabel('Gain (dB)')
pylab.ylim(-120., 3.)
pylab.grid(True)

if args.sos:
    w, h = scipy.signal.sosfreqz(sos, worN=8000, fs=samplerate)
    pylab.plot(w, 20 * numpy.log10(numpy.absolute(h)), linewidth=2)
else:
    w, h = scipy.signal.freqz(b, a, worN=8000)
    pylab.plot(
        (w / numpy.pi) * nyquist,
        20 * numpy.log10(numpy.absolute(h)),
        linewidth=2)

#------------------------------------------------
# Plot the group delay of the filter.
#------------------------------------------------
pylab.figure(2)
pylab.title('Group Delay')
pylab.xlabel('Frequency (Hz)')
pylab.ylabel('Samples')
pylab.grid(True)

if args.sos:
    # Convert SOS to transfer function for group delay calculation
    b, a = scipy.signal.sos2tf(sos)

w, gd = scipy.signal.group_delay((b, a))
pylab.plot(
    (w / numpy.pi) * nyquist, numpy.absolute(gd), linewidth=2)

pylab.show()
