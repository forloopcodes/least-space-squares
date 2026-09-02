"""Best known side lengths s(n) from the literature (reference data for benchmarking).

Source: "Squares in Squares" by Erich Friedman, maintained with high-precision updates by
David Ellsworth (https://kingbird.myphotos.cc/packing/squares_in_squares.html), snapshot of
September 2026; originally E. Friedman, "Packing Unit Squares in Squares: A Survey and New
Results", Electronic Journal of Combinatorics, Dynamic Survey DS7.

Only the n with a non-trivial record are listed; for every other n <= 324 the best known packing is
the trivial one, s = ceil(sqrt(n)).  Removing squares from a packing keeps it valid, so the best known
value for an unlisted n is the minimum over listed m >= n (see :func:`best_known`).
Entries: n -> (s, closed form or polynomial degree, attribution).  "Proved" means optimality is proven.
"""
from __future__ import annotations
import math

RECORDS = {
    1: (1.0, '1', 'Trivial.'),
    2: (2.0, '2', 'Proved by Frits Göbel in early 1979. --> − n − 1) pattern Rigid packings'),
    3: (2.0, '2', 'Proved by Frits Göbel in early 1979.'),
    4: (2.0, '2', 'Trivial.'),
    5: (2.70710678118654, '2 + 1/2sqrt 2', 'Rigid. Proved by Frits Göbel in early 1979.'),
    6: (3.0, '3', 'Proved by Michael Kearney and Peter Shiu in June 2001.'),
    7: (3.0, '3', 'Proved by Erich Friedman in 1999. -->'),
    8: (3.0, '3', 'Proved by Erich Friedman in 1999.'),
    9: (3.0, '3', 'Trivial.'),
    10: (3.70710678118654, '3 + 1/2sqrt 2', 'Found by Frits Göbel in early 1979. Proved by Walter Stromquist in 2003.'),
    11: (3.87708359002281, 'deg8-poly', 'Rigid. Found by Walter Trump in 1979.'),
    13: (4.0, '4', 'Proved by Wolfram Bentz in August 2009.'),
    14: (4.0, '4', 'Proved by Erich Friedman in 1999. -->'),
    15: (4.0, '4', 'Proved by Erich Friedman in 1999.'),
    17: (4.67553009360455, 'deg18-poly', 'Found by John Bidwell in 1998. Based on packing found by Pertti Hämäläinen in 1980.'),
    18: (4.82287565553229, '7/2 + 1/2sqrt 7', 'Found by Pertti Hämäläinen in 1980. Pictured alternative with minimal rotated squares found by Mats Gustafsson'),
    19: (4.88561808316412, '3 + 4/3sqrt 2', 'Found first by Robert Wainwright in late 1979. Based on packing found by Charles F. Cottingham in early 1979.'),
    22: (5.0, '5', 'Proved by Wolfram Bentz in October 2018.'),
    23: (5.0, '5', 'Proved by Hiroshi Nagamochi in 2005.'),
    24: (5.0, '5', 'Proved by Erich Friedman in 1999.'),
    26: (5.62132034355964, '7/2 + 3/2sqrt 2', 'Found by Erich Friedman in 1997. Unextends the found by Evert Stenlund in early 1980.'),
    27: (5.70710678118654, '5 + 1/2sqrt 2', 'Found by Frits Göbel in early 1979.'),
    28: (5.82444461667405, 'deg6-poly', "Rigid. Found by David Ellsworth in December 2025, using his modified version of Thomas Schadt's simulated anne"),
    29: (5.93383346267692, 's', 'Found by Thomas Schadt in December 2025, using a simulated annealing program he wrote, starting from randomnes'),
    33: (6.0, '6', 'Proved by Wolfram Bentz in October 2018.'),
    34: (6.0, '6', 'Proved by Hiroshi Nagamochi in 2005.'),
    35: (6.0, '6', 'Proved by Erich Friedman in 1999.'),
    37: (6.59861960924436, 'deg8-poly', 'Found by David W. Cantrell in September 2002. Improves upon the found by Evert Stenlund in early 1980.'),
    38: (6.70710678118654, '6 + 1/2sqrt 2', 'Found by Frits Göbel in early 1979.'),
    39: (6.81072208306864, 'deg5-poly', 'Found by Thomas Schadt in January 2026, using a simulated annealing program he wrote, starting from randomness'),
    40: (6.82842712474619, '4 + 2 sqrt 2', 'Rigid. Found by Frits Göbel in early 1979.'),
    41: (6.9266930944688, 'deg42-poly', 'Found by Thomas Schadt in December 2025, using a simulated annealing program he wrote, starting from randomnes'),
    46: (7.0, '7', 'Proved by Wolfram Bentz in August 2009.'),
    47: (7.0, '7', 'Proved by Hiroshi Nagamochi in 2005. -->'),
    48: (7.0, '7', 'Proved by Hiroshi Nagamochi in 2005.'),
    50: (7.57142857142857, '7 + 4/7', 'Found by Thomas Schadt in December 2025, using a simulated annealing program he wrote, starting from randomnes'),
    51: (7.70079923541701, 'deg12-poly', 'Found by Thomas Schadt in January 2026, using a simulated annealing program he wrote, starting from randomness'),
    52: (7.70710678118654, '7 + 1/2sqrt 2', 'Found by Frits Göbel in early 1979.'),
    53: (7.82287565553229, '13/2 + 1/2sqrt 7', 'Found by David W. Cantrell in September 2002. Improved by David W. Cantrell in December 2024. Improved by Davi'),
    54: (7.84666719284348, '7-1/2sqrt 2+sqrt{1+sqrt 2}', 'Found by David W. Cantrell in October 2005. Improved by Joe DeVincentis in April 2014.'),
    55: (7.94577100750391, 's', 'Found by Thomas Schadt in January 2026, using a simulated annealing program he wrote, starting from a cherry-p'),
    62: (8.0, '8', 'Proved by Hiroshi Nagamochi in 2005. -->'),
    63: (8.0, '8', 'Proved by Hiroshi Nagamochi in 2005.'),
    65: (8.53553390593273, '5 + 5/2sqrt 2', 'Found by Frits Göbel in early 1979.'),
    66: (8.65685424949238, '3 + 4 sqrt 2', 'Found by Evert Stenlund in early 1980.'),
    67: (8.70710678118654, '8 + 1/2sqrt 2', 'Found by Evert Stenlund in early 1980, extending the found by Frits Göbel in early 1979.'),
    68: (8.80345993651653, 's', 'Found by Sigvart Brendberg in June 2023, using a computer program he wrote followed by manual optimization. Im'),
    69: (8.827212055929, 'deg82-poly', 'Found by Maurizio Morandi in June 2010. Improved by David W. Cantrell in August 2023.'),
    70: (8.881666757009, 'deg4-poly', 'Found by Joe DeVincentis in April 2014.'),
    71: (8.94407155757031, 's', 'Found by Thomas Schadt in December 2025, using a simulated annealing program he wrote, improving upon the foun'),
    79: (9.0, '9', 'Proved by Hiroshi Nagamochi in 2005. -->'),
    80: (9.0, '9', 'Proved by Hiroshi Nagamochi in 2005.'),
    82: (9.53553390593273, '6 + 5/2sqrt 2', 'Found by Frits Göbel in early 1979. Adds two "L"s to .'),
    83: (9.63482562092335, 'deg24-poly', 'Found by Károly Hajba in September 2024. Improved upon the found by Evert Stenlund in early 1980. Improved by '),
    84: (9.70710678118654, '9 + 1/2sqrt 2', 'Found by Evert Stenlund in early 1980, extending the found by Frits Göbel in early 1979.'),
    85: (9.74264068711928, '11/2 + 3 sqrt 2', 'Found by Erich Friedman in 1997.'),
    86: (9.82287565553229, '17/2 + 1/2sqrt 7', 'Found by Erich Friedman in 1997. Extends the alternative packing of the found by Pertti Hämäläinen in 1980 fou'),
    87: (9.83881743996618, 'deg44-poly', 'Found by David Ellsworth in December 2024, based on the found by Károly Hajba in November 2024 and the found b'),
    88: (9.88815305375857, 'deg20-poly', 'Found by David Ellsworth in November 2024, by adapting and extending the improvement found by David W. Cantrel'),
    89: (9.94974746830583, '5 + 7/2sqrt 2', 'Found by Evert Stenlund in early 1980, by continuing a pattern found by Frits Göbel in early 1979.'),
    98: (10.0, '10', 'Proved by Hiroshi Nagamochi in 2005. -->'),
    99: (10.0, '10', 'Proved by Hiroshi Nagamochi in 2005.'),
    101: (10.53553390593273, '7 + 5/2sqrt 2', 'Adds two "L"s to the found by Frits Göbel in early 1979.'),
    102: (10.61138823373863, 'deg8-poly', 'Found by Károly Hajba in September 2024. Extended the found by Evert Stenlund in early 1980. Improved by David'),
    103: (10.70383477210707, 's', 'Found by Thomas Schadt in December 2025, using a simulated annealing program he wrote, starting from randomnes'),
    104: (10.70710678118654, '10 + 1/2sqrt 2', 'Extends the found by Frits Göbel in early 1979.'),
    105: (10.80789399144854, 's', 'Found by Thomas Schadt in December 2025, using a simulated annealing program he wrote, starting from randomnes'),
    106: (10.82297973416944, 'deg32-poly', 'Found by David Ellsworth in November 2024, based on the Improved by David W. Cantrell in December 2024.'),
    107: (10.84666719284348, '10-1/2sqrt 2+sqrt{1+sqrt 2}', 'Found by Károly Hajba in November 2024. Is an alternative packing for an extension of the found by Joe DeVince'),
    108: (10.92591939016138, 'deg144-poly', 'Found by Károly Hajba in October 2024. Improved by David Ellsworth in November 2024. Extends the found by Walt'),
    109: (10.94974746830583, '6 + 7/2sqrt 2', 'Continues a pattern found by Frits Göbel in early 1979.'),
    110: (10.99683777797875, 's', 'Found by David W. Cantrell in February 2025. Bounds the conjecture to . Improved by David Ellsworth in January'),
    119: (11.0, '11', 'Proved by Hiroshi Nagamochi in 2005. -->'),
    120: (11.0, '11', 'Proved by Hiroshi Nagamochi in 2005.'),
    122: (11.53553390593273, '8 + 5/2sqrt 2', 'Adds three "L"s to the found by Frits Göbel in early 1979.'),
    123: (11.60139979378801, 'deg12-poly', 'Found and improved by David Ellsworth in December 2024, by extending the found by Károly Hajba in September 20'),
    124: (11.65685424949238, '6 + 4 sqrt 2', 'Continues a pattern found by Frits Göbel in early 1979.'),
    125: (11.70710678118654, '11 + 1/2sqrt 2', 'Extends the found by Frits Göbel in early 1979.'),
    126: (11.77617894651987, 'deg59-poly', 'Found by David Ellsworth in December 2024, based on the Improved by David W. Cantrell in December 2024. Improv'),
    127: (11.82287565553229, '21/2 + 1/2sqrt 7', 'Extends the found by Erich Friedman in 1997.'),
    128: (11.82509196821368, 'deg40-poly', 'Found by David Ellsworth in November 2024, based on the found by Maurizio Morandi in June 2010. Improved by Da'),
    129: (11.88130621809, 'deg20-poly', 'Found and improved by David Ellsworth in December 2024, adapting/extending the improvement found by David W. C'),
    130: (11.91119052015898, 'deg8-poly', 'Found by David Ellsworth in November 2024. Improved by David W. Cantrell in November 2024. Improved by David E'),
    131: (11.95654869347733, 's', "Found by David Ellsworth in January 2026, using his modified version of Thomas Schadt's simulated annealing pr"),
    132: (11.99143643966336, 's', 'Found by M.Z. Arslanov, S.A. Mustafin, and Z.K. Shangitbayev in March 2019. Bounded the conjecture to . Improv'),
    142: (12.0, '12', 'Proved by Hiroshi Nagamochi in 2005. -->'),
    143: (12.0, '12', 'Proved by Hiroshi Nagamochi in 2005.'),
    145: (12.53553390593273, '9 + 5/2sqrt 2', 'Adds four "L"s to the found by Frits Göbel in early 1979.'),
    146: (12.60090777851301, 'deg16-poly', 'Found by David W. Cantrell in January 2025 by switching to the rotationally symmetric form of adding an "L" to'),
    148: (12.65685424949238, '7 + 4 sqrt 2', 'Continues a pattern found by Frits Göbel in early 1979.'),
    149: (12.70710678118654, '12 + 1/2sqrt 2', 'Extends the found by Frits Göbel in early 1979.'),
    150: (12.77817459305202, '5 + 11/2sqrt 2', 'Found by David Ellsworth in January 2025. Extends the found by Erich Friedman in 1997 how the found by Evert S'),
    151: (12.82287565553229, '23/2 + 1/2sqrt 7', 'Found by David Ellsworth in November 2024. Improved by David W. Cantrell in December 2024.'),
    152: (12.83100282216725, 'deg84-poly', 'Found by David Ellsworth and David W. Cantrell in January 2025, based on the found by David W. Cantrell in Sep'),
    153: (12.881666757009, 'deg4-poly', 'Found by David Ellsworth in November 2024, based on the found by Joe DeVincentis in April 2014.'),
    154: (12.93171183926903, 's', 'Originally found by David Ellsworth in December 2024, by combining two slightly modified copies of the found b'),
    155: (12.9585138860669, 's', "Found by David Ellsworth in January 2026, using his modified version of Thomas Schadt's simulated annealing pr"),
    156: (12.982191723548, 's', 'Found by M.Z. Arslanov, S.A. Mustafin, and Z.K. Shangitbayev in March 2019. Shows for . Improved by David Ells'),
    167: (13.0, '13', 'Proved by Hiroshi Nagamochi in 2005. -->'),
    168: (13.0, '13', 'Proved by Hiroshi Nagamochi in 2005.'),
    170: (13.53553390593273, '10 + 5/2sqrt 2', 'Extends the found by Frits Göbel in early 1979. This alternative, converting the augmented by five "L"s into a'),
    171: (13.57142857142857, '13 + 4/7', 'Found by David Ellsworth in December 2025, by combining two copies of the found by Thomas Schadt in December 2'),
    172: (13.6189889866016, 'deg8-poly', 'Found by Károly Hajba in November 2024, extending the he found in September 2024. Improved by David Ellsworth '),
    173: (13.65685424949238, '8 + 4 sqrt 2', 'Adds an "L" to the that continues a pattern found by Frits Göbel in early 1979.'),
    174: (13.70710678118654, '13 + 1/2sqrt 2', 'Extends the found by Frits Göbel in early 1979.'),
    175: (13.77817459305202, '6 + 11/2sqrt 2', 'Found by David Ellsworth in December 2024. Based on the that continues a pattern found by Frits Göbel in early'),
    176: (13.82287565553229, '25/2 + 1/2sqrt 7', 'Extends the found by Erich Friedman in 1997.'),
    177: (13.82302875075647, 'deg32-poly', 'Found and improved by David Ellsworth in November 2024 and December 2024, based on the found/improved by David'),
    178: (13.84666719284348, '13-1/2sqrt 2+sqrt{1+sqrt 2}', 'Extends the found by Joe DeVincentis in April 2014.'),
    179: (13.8954098224364, '25/2 + sqrt 2', 'Found by David Ellsworth in January 2025, using a computer program he wrote. Improved by David Ellsworth in Ja'),
    180: (13.93513929847193, 's', "Found by David Ellsworth in January 2026, using his modified version of Thomas Schadt's simulated annealing pr"),
    181: (13.95698416446504, 's', "Found by David Ellsworth in January 2026, using his modified version of Thomas Schadt's simulated annealing pr"),
    182: (13.97442960739443, 's', 'Found by M.Z. Arslanov, S.A. Mustafin, and Z.K. Shangitbayev in March 2019. Shows for . Improved by David Ells'),
    195: (14.0, '14', 'Proved by Hiroshi Nagamochi in 2005.'),
    197: (14.53553390593273, '11 + 5/2sqrt 2', 'Adds six "L"s to the found by Frits Göbel in early 1979.'),
    198: (14.57142857142857, '14 + 4/7', 'Adds an "L" to the found by David Ellsworth in December 2025 by combining two copies of the found by Thomas Sc'),
    199: (14.6189889866016, 'deg8-poly', 'Adds an "L" to the found by Károly Hajba in November 2024, improved by David Ellsworth in November 2024 by ada'),
    200: (14.65685424949238, '9 + 4 sqrt 2', 'Adds two "L"s to the that continues a pattern found by Frits Göbel in early 1979.'),
    201: (14.70710678118654, '14 + 1/2sqrt 2', 'Extends the found by Frits Göbel in early 1979.'),
    202: (14.72792206135785, '2 + 9 sqrt 2', 'Extends the found by Frits Göbel in early 1979.'),
    203: (14.77817459305202, '7 + 11/2sqrt 2', 'Found by David Ellsworth in December 2024. Based on the that continues a pattern found by Frits Göbel in early'),
    204: (14.82287565553229, '27/2 + 1/2sqrt 7', 'Extends the found by Erich Friedman in 1997.'),
    205: (14.82445114612408, 'deg40-poly', 'Found by David Ellsworth in December 2024, by extending the he found/improved in November/December 2024, based'),
    206: (14.87253189075152, 's', 'Found and improved by David Ellsworth in December 2024, adapting/extending the improvement found by David W. C'),
    207: (14.8939785956378, 's', 'Found by David Ellsworth in November 2024, by extending the found by Erich Friedman in 1997, and adapting and '),
    208: (14.93783044811097, 's', "Found by David Ellsworth in January 2026, using his modified version of Thomas Schadt's simulated annealing pr"),
    209: (14.9586824407826, 's', "Found by David Ellsworth in January 2026, using his modified version of Thomas Schadt's simulated annealing pr"),
    210: (14.97421396826961, 's', 'Found by M.Z. Arslanov, S.A. Mustafin, and Z.K. Shangitbayev in March 2019. Shows for . Improved by David Ells'),
    224: (15.0, '15', 'Proved by Hiroshi Nagamochi in 2005.'),
    226: (15.53553390593273, '12 + 5/2sqrt 2', 'Adds seven "L"s to the found by Frits Göbel in early 1979.'),
    227: (15.57106781186547, '17/2 + 5 sqrt 2', 'Found by David Ellsworth in January 2025, using a computer program he wrote. Continues the , series found by E'),
    228: (15.60902282132495, 'deg12-poly', 'Found by David Ellsworth in November 2024, by extending the found by Károly Hajba in September 2024 and improv'),
    229: (15.65685424949238, '10 + 4 sqrt 2', 'Adds three "L"s to the that continues a pattern found by Frits Göbel in early 1979.'),
    230: (15.68292682926829, '15 + 28/41', 'Found and improved by David Ellsworth in January 2025. Uses a rotational symmetry technique found by David W. '),
    231: (15.70710678118654, '15 + 1/2sqrt 2', 'Extends the found by Frits Göbel in early 1979.'),
    233: (15.77817459305202, '8 + 11/2sqrt 2', 'Continues a pattern found by Frits Göbel in early 1979.'),
    234: (15.82287565553229, '29/2 + 1/2sqrt 7', 'Found by David Ellsworth in December 2024, by extending the and he found/improved in November/December 2024, b'),
    235: (15.82660563342856, 'deg83-poly', 'Found by David Ellsworth and David W. Cantrell in January 2025, based on the found by David W. Cantrell in Sep'),
    236: (15.87607676541001, 'deg12-poly', 'Found by David Ellsworth in November 2024, by extending the found by Erich Friedman in 1997 and adapting the i'),
    237: (15.91421356237309, '29/2 + sqrt 2', 'Found by David Ellsworth in December 2024. Similar to the found by Erich Friedman in 1997.'),
    238: (15.93984676308969, 's', "Found by David Ellsworth in February 2026, using his modified version of Thomas Schadt's simulated annealing p"),
    239: (15.95643304058435, 's', "Found by David Ellsworth in January 2026, using his modified version of Thomas Schadt's simulated annealing pr"),
    240: (15.97559404379946, 's', 'Originally found by Károly Hajba in September 2015. Bounded the conjecture to . Converted in December 2024 fro'),
    241: (15.99091684780193, 's', 'Found by M.Z. Arslanov, S.A. Mustafin, and Z.K. Shangitbayev in March 2019. Shows for . Improved by David Ells'),
    255: (16.0, '16', 'Proved by Hiroshi Nagamochi in 2005.'),
    257: (16.53553390593273, '13 + 5/2sqrt 2', 'Extends the found by Frits Göbel in early 1979. This alternative, converting the augmented by eight "L"s into '),
    258: (16.57106781186547, '19/2 + 5 sqrt 2', 'Adds an "L" to the found by David Ellsworth in January 2025 (using a computer program he wrote) which continue'),
    259: (16.60257141234448, 'deg8-poly', 'Found by David Ellsworth in December 2024, by extending the found by Károly Hajba in September 2024 and adapti'),
    260: (16.65685424949238, '11 + 4 sqrt 2', 'Extends the that continues a pattern found by Frits Göbel in early 1979. This alternative, converting the augm'),
    261: (16.68292682926829, '16 + 28/41', 'Adds an "L" to the found and improved by David Ellsworth in January 2025, which uses a rotational symmetry tec'),
    262: (16.70710678118654, '16 + 1/2sqrt 2', 'Extends the found by Frits Göbel in early 1979.'),
    263: (16.74264068711928, '25/2 + 3 sqrt 2', 'Found by David Ellsworth in January 2025, based on the he found.'),
    265: (16.77817459305202, '9 + 11/2sqrt 2', 'Continues a pattern found by Frits Göbel in early 1979.'),
    266: (16.8230620828378, 'deg32-poly', 'Found and improved by David Ellsworth in November 2024 and in December 2024, based on the found/improved by Da'),
    267: (16.84666719284348, '16-1/2sqrt 2+sqrt{1+sqrt 2}', 'Extends the found by Károly Hajba in November 2024.'),
    268: (16.87933209237563, 'deg6-poly', 'Found by David Ellsworth in November 2024, based on the found by Joe DeVincentis in April 2014. Improved by Da'),
    269: (16.90596764828402, 'deg8-poly', 'Found by David Ellsworth in November 2024, by extending the found by Erich Friedman in 1997, and adapting and '),
    270: (16.94073593015457, 's', "Found by David Ellsworth in February 2026, using his modified version of Thomas Schadt's simulated annealing p"),
    271: (16.95509447506437, 's', "Found by David Ellsworth in February 2026, using his modified version of Thomas Schadt's simulated annealing p"),
    272: (16.96980702828259, 's', 'Originally found by Lars Cleemann between 1991 and 1998. Bounded the conjecture to . Converted in December 202'),
    273: (16.98832058897683, 's', 'Found by M.Z. Arslanov, S.A. Mustafin, and Z.K. Shangitbayev in March 2019. Shows for . Improved by David Ells'),
    288: (17.0, '17', 'Proved by Hiroshi Nagamochi in 2005.'),
    291: (17.53553390593273, '14 + 5/2sqrt 2', 'Combines two copies of the that continues a pattern found by Frits Göbel in early 1979. This is the first that'),
    292: (17.60257141234448, 'deg8-poly', 'Adds an "L" to the found and improved by David Ellsworth in December 2024 extending the found by Károly Hajba '),
    293: (17.63414634146341, '17 + 26/41', 'Found by David Ellsworth in December 2024. Improved by David W. Cantrell in January 2025. This is the first re'),
    294: (17.65685424949238, '12 + 4 sqrt 2', 'Extends the that continues a pattern found by Frits Göbel in early 1979. This is likely an irreducible primiti'),
    296: (17.70710678118654, '17 + 1/2sqrt 2', 'Extends the found by Frits Göbel in early 1979.'),
    297: (17.74116992948972, 's', 'Found by David Ellsworth in January 2025. Extends the found by Erich Friedman in 1997. Improved by David Ellsw'),
    298: (17.77817459305202, '10 + 11/2sqrt 2', 'Adds an "L" to the that continues a pattern found by Frits Göbel in early 1979.'),
    299: (17.82287565553229, '33/2 + 1/2sqrt 7', 'Extends the found by Erich Friedman in 1997.'),
    300: (17.82412338847854, 'deg40-poly', 'Found by David Ellsworth in January 2025, extending the and he found, based on the found by Maurizio Morandi i'),
    301: (17.86899185179999, 's', 'Drafted by David Ellsworth in January 2025, including adapting and extending the improvement found by David W.'),
    302: (17.88674602860566, 'deg4-poly', 'Found by David Ellsworth in December 2024 (including re-adapting the techniques from the and / he improved), b'),
    303: (17.93127894394689, 's', "Found by David Ellsworth in April 2026, using his modified version of Thomas Schadt's simulated annealing prog"),
    304: (17.9491720191911, 's', "Found by David Ellsworth in May 2026, using his modified version of Thomas Schadt's simulated annealing progra"),
    305: (17.96075558628675, 's', "Found by David Ellsworth in March 2026, using his modified version of Thomas Schadt's simulated annealing prog"),
    306: (17.96926975248972, 's', "Found by David Ellsworth in February 2026, using his modified version of Thomas Schadt's simulated annealing p"),
    307: (17.98281564631754, 's', 'Found by M.Z. Arslanov, S.A. Mustafin, and Z.K. Shangitbayev in March 2019. Shows for . Improved by David Ells'),
    323: (18.0, '18', 'Proved by Hiroshi Nagamochi in 2005.'),
    626: (25.5208152801713, '27/2 + 17/2sqrt 2', 'Found by David Ellsworth in January 2025, using a computer program he wrote. Continues the , series found by E'),
}


def best_known(n: int):
    """Best known side length for ``n`` unit squares (None if beyond the table, n > 324)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    best = math.ceil(math.sqrt(n))
    if n > 324:
        return None
    for m, (s, _ex, _note) in RECORDS.items():
        if m >= n and s < best:
            best = s
    return float(best)


def _grouped_with_next(n: int):
    """The source page labels the proven pairs (k^2-2, k^2-1) as one entry ("194, 195 ... Proved");
    the table keeps the larger n, so the smaller one inherits that entry."""
    m = n + 1
    k = math.isqrt(m + 1)
    if m in RECORDS and (k * k - 1 == m) and RECORDS[m][0] == math.ceil(math.sqrt(n)) and "Proved" in RECORDS[m][2]:
        return m
    return None


def record_entry(n: int):
    """The record that the best-known value for ``n`` derives from: (m, s, exact, note)."""
    best = best_known(n)
    if best is None:
        return None
    g = _grouped_with_next(n)
    if n not in RECORDS and g is not None:
        s, ex, note = RECORDS[g]
        return n, s, ex, note
    for m, (s, ex, note) in sorted(RECORDS.items()):
        if m >= n and abs(s - best) < 1e-12:
            return m, s, ex, note
    return n, float(math.ceil(math.sqrt(n))), str(math.ceil(math.sqrt(n))), "trivial grid"


def is_proved(n: int) -> bool:
    """True when the best known value is proven optimal (as recorded in the table notes)."""
    ent = record_entry(n)
    if ent is None:
        return False
    m, s, ex, note = ent
    if m == n and ("Proved" in note or "Trivial" in note):
        return True
    if _grouped_with_next(n) is not None:
        return True
    k = math.isqrt(n)
    # n = k^2, k^2 - 1, k^2 - 2 are proven (Nagamochi 2005) for all k; small cases are in the table.
    if k * k == n or (k + 1) ** 2 - n in (1, 2):
        return True
    return False
