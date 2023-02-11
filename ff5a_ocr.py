#! python3
# coding: utf-8

try:
    from cnocr import CnOcr
except:
    print('''
please install Pillow with
pip3 install cnocr
or
pip install cnocr
or
py -3 -m pip install cnocr
or
python -m pip install cnocr
''')
    raise

def report(*args):
    r = ' '.join(args)
    print(r)
    return r

class c_map_guesser:

    MAX_LA_WARN = 3
    MAX_LA_SKIP = 6

    def __init__(self):
        self.det = {}
        self.det_r = {}
        self.nondet = {}
        self.cnflct = {}

    def innate(self, knowledge):
        for c1, c2 in knowledge.items():
            self.det[c1] = c2
            self.det_r[c2] = c1

    def _norm_text(self, s, norm, trim):
        r = []
        for c in s:
            if c in trim:
                continue
            if c in norm:
                c = norm[c]
            r.append(c)
        return r

    def _log_conflict(self, c1, i1, c2, i2, cmt):
        report('info', f'conflict {c1} - {c2} {i1}/{i2} at {cmt}')
        if c1 in self.cnflct:
            cc_info = self.cnflct[c1]
        else:
            cc_info = {}
            self.cnflct[c1] = cc_info
        cc_info[c2] = (cmt, i1, i2)

    def _guess_match(self, c1, i1, c2, i2, cmt):
        if c1 in self.det:
            if c2 == self.det[c1]:
                return
            self._log_conflict(c1, i1, c2, i2, cmt)
            return
        if c2 in self.det_r:
            assert c1 != self.det_r[c2]
            self._log_conflict(c1, i1, c2, i2, cmt)
            return
        if not c1 in self.nondet:
            self.nondet[c1] = {}
        c1r_info = self.nondet[c1]
        if c2 in c1r_info:
            self.det[c1] = c2
            self.det_r[c2] = c1
            del self.nondet[c1]
            for cc2, (ccmt, ci1, ci2) in c1r_info.items():
                if cc2 == c2:
                    continue
                self._log_conflict(c1, ci1, cc2, ci2, ccmt)
        else:
            c1r_info[c2] = (cmt, i1, i2)

    def _guess_match_blk(self, b1, i1, b2, i2, cmt):
        print('guess block', cmt)
        print(' '.join(hex(c)[2:] for c in b1))
        print(''.join(b2))

    def feed(self, s1, s2, cmt, norm_r = {}, trim_r = []):
        trim1 = set()
        trim2 = set()
        for t in trim_r:
            if t in self.det_r:
                trim1.add(self.det_r[t])
                trim2.add(t)
        s1 = self._norm_text(s1, {}, trim1)
        s2 = self._norm_text(s2, norm_r, trim2)
        print('feed', cmt, ''.join(s2))
        #print(' '.join(hex(c)[2:] for c in s1))
        l1 = len(s1)
        l2 = len(s2)
        i1 = 0
        i2 = 0
        sk1 = []
        sk2 = []
        lst_matched = True
        while i1 < l1 and i2 < l2:
            if lst_matched and sk1 and sk2:
                _lsk1 = len(sk1)
                _lsk2 = len(sk2)
                _i1 = i1 - _lsk1
                _i2 = i2 - _lsk2
                for _ in range(min(_lsk1, _lsk2)):
                    _c1 = s1[_i1]
                    _c2 = s2[_i2]
                    self._guess_match(_c1, _i1, _c2, _i2, cmt)
                    _i1 += 1
                    _i2 += 1
##                self._guess_match_blk(
##                    sk1, i1 - len(sk1), sk2, i2 - len(sk2), cmt)
                sk1 = []
                sk2 = []
            c1 = s1[i1]
            c2 = s2[i2]
            if c1 in self.det:
                # known c1
                c1r = self.det[c1]
                if c1r == c2:
                    # matched, bypass
                    i1 += 1
                    i2 += 1
                    #print('matched', c2)
                    lst_matched = True
                    continue
                # find matched char in s2 next
                _i2dlt = 0
                _sk2 = [c2]
                for _i2 in range(i2 + 1, min(l2, i2 + self.MAX_LA_SKIP)):
                    _c2 = s2[_i2]
                    if _c2 == c1r:
                        _i2dlt = _i2 - i2
                        break
                    _sk2.append(_c2)
                if _i2dlt > 0:
                    if _i2dlt >= self.MAX_LA_WARN:
                        report('warning',
                            f's1 lookahead too mush {i2}+{_i2dlt}/{self.MAX_LA_WARN}~{self.MAX_LA_SKIP} at {cmt}')
                    i1 += 1
                    i2 += _i2dlt + 1
                    #print('sk2+1', ''.join(_sk2))
                    sk2.extend(_sk2)
                    lst_matched = True
                    continue
                # no matched char found, skip c1
                #print('sk1+2')
                sk1.append(c1)
                i1 += 1
                lst_matched = False
                continue
            # unknown c1
            if c2 in self.det_r:
                # known c2
                c2r = self.det_r[c2]
                assert c2r != c1
                # find matched char in s1 next
                _i1dlt = 0
                _sk1 = [c1]
                for _i1 in range(i1 + 1, min(l1, i1 + self.MAX_LA_SKIP)):
                    _c1 = s1[_i1]
                    if _c1 == c2r:
                        _i1dlt = _i1 - i1
                        break
                    _sk1.append(_c1)
                if _i1dlt > 0:
                    if _i1dlt >= self.MAX_LA_WARN:
                        report('warning',
                            f's2 lookahead too mush {i1}+{_i1dlt}/{self.MAX_LA_WARN}~{self.MAX_LA_SKIP}')
                    i2 += 1
                    i1 += _i1dlt + 1
                    #print('sk1+1')
                    sk1.extend(_sk1)
                    lst_matched = True
                    continue
                # no matched char found, skip c2
                #print('sk2+2', c2)
                sk2.append(c2)
                i2 += 1
                lst_matched = False
                continue
            # both c1 c2 unknown
##            sk1.append(c1)
##            #print('sk2+3', c2)
##            sk2.append(c2)
##            lst_matched = False
            self._guess_match(c1, i1, c2, i2, cmt)
            i1 += 1
            i2 += 1

class c_ff5a_ocr_parser:

    def __init__(self, txt_parser):
        self.tpsr = txt_parser

    def parse(self):
        self.ocr = CnOcr(det_model_name='naive_det')

    def draw_chars(self, chars, pad = 3):
        blk = self.tpsr.draw_chars(chars, pad_col = pad)
        return self.tpsr.draw.make_img(blk)

    def ocr_chars(self, chars, ret_img = False):
        im = self.draw_chars(chars)
        rinfo = self.ocr.ocr(im)
        rchars = ''.join(i['text'] for i in rinfo)
        if ret_img:
            return rchars, im
        else:
            return rchars

    def pick_text(self, tidxs, trims = [], txt = None):
        if txt is None:
            txt = []
        for tidx in tidxs:
            t = self.tpsr.get_text(tidx)
            for c in t:
                if self.tpsr.is_ctrl(c):
                    continue
                for trim_st, trim_ed in trims:
                    if trim_st <= c < trim_ed:
                        break
                else:
                    txt.append(c)
        return txt

if __name__ == '__main__':

    from pprint import pprint
    ppr = lambda *a, **ka: pprint(*a, **ka, sort_dicts = False)
    
    from ff5a_parser import c_ff5a_parser
    
    def main():
        psr = c_ff5a_parser('ff5acn.gba')
        psr.load_rom()
        psr.parse()
        ocr = c_ff5a_ocr_parser(psr.txt_parser['cn'])
        ocr.parse()
        return ocr
    ocr = main()
    def init_guesser():
        gsr = c_map_guesser()
        gsr.innate({
            0x2c: ',',
            0x2e: '.',
            0x21: '!',
            0x3f: '?',
            0x95: '「',
            0x96: '」',
            0x91: '…',
            0x92: ' ',
            **{i: chr(i) for i in range(0x30, 0x3a)},
        })
        norm = {
            '，': ',',
            '。': '.',
            '？': '?',
            '！': '!',
        }
        trim = [' ', '…']
        trim_rng = [(0x99, 0x13b)]
        #for i in range(1800, 1820, 20):
        for i in range(1800, 1900, 20):
            stxt = ocr.pick_text(range(i + 1, i + 21, 2), trim_rng)
            rtxt = ocr.ocr_chars(stxt)
            gsr.feed(stxt, rtxt, i, norm, trim)
        return gsr
    gsr = init_guesser()
    print(''.join([*gsr.det.values()][15:50]))
    im = ocr.draw_chars([k for k in gsr.det][15:50])
    
