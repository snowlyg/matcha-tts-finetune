#!/usr/bin/env python3
"""生成播报扩展录制稿（spoken 形式：数字全写汉字、字母逐个念）。

覆盖 v1 自采样实测的高危结构：小数体温、轻声功能词链（为您/请您/接一下）、
双字名词+数量（毫升/片/支）、动宾紧凑（上报值班/携带就诊卡）、姓名+长句、
字母科室（I C U 读字母名）、电话/床号数字串（幺/零读法）。

用法：
  python scripts/make_record_script.py --n 1000 --out data/record_script_ext.tsv
"""
import argparse
import collections
import random

DIGITS = "零一二三四五六七八九"
YAO = "幺"

SURNAMES = list("张王李赵刘陈杨黄周吴徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛段雷侯龙史陶黎贺顾毛郝龚邵万钱严戴莫孔向汤")
GIVEN = ["三", "四", "五", "伟", "芳", "娜", "敏", "静", "丽", "强", "磊", "军", "洋", "勇", "艳",
         "杰", "娟", "涛", "明", "超", "霞", "平", "刚", "建国", "国强", "秀英", "丽华",
         "桂芳", "玉梅", "志强", "建华", "小红", "小明", "国庆", "淑珍", "凤英", "德福", "春梅"]

DEPTS = ["护士站", "医生办公室", "治疗室", "处置室", "配液中心", "缴费处", "检验科", "影像科",
         "采血室", "输液室", "药房窗口", "住院部大厅"]
ACTIONS = ["呼叫护士", "正在呼叫", "请按呼叫铃确认", "有新的呼叫请求", "请到{d}",
           "请在床边稍候", "请返回病房等待", "请到{d}等候", "输液即将完成", "输液已完成",
           "雾化即将结束", "心电监护已解除", "治疗即将开始", "患者需要帮助", "请准备接收"]
PROC = ["输液", "雾化", "抽血", "心电监护", "吸氧", "换药", "打针", "量血压", "测血糖", "拍片",
        "做彩超", "灌肠", "导尿", "皮试", "拍胸片"]
LETTERS = ["I C U", "C C U", "N I C U", "P I C C", "J H C", "L I S", "O R", "E N T",
           "P A C S", "R I C U", "A", "B", "C", "D", "E"]
LETTER_OBJ = ["{b}患者", "{b}", "主机", "标本", "维护", "手术室", "导管室"]
TIMES_M = ["上午", "下午", "中午", "晚上", "凌晨", "清晨"]
MINUTES = ["整", "零五分", "十分", "十五分", "二十分", "二十五分", "半", "三十五分", "四十分", "四十五分", "五十分", "五十五分"]
REMIND = ["请保持安静", "请勿倚靠电梯门", "请携带就诊卡", "注意防滑", "请按规定时间探视",
          "注意用电安全", "请保管好个人物品", "请勿在病房内吸烟", "请留意叫号屏幕",
          "请配合护士工作", "注意地面湿滑", "请看护好小孩"]
# 轻声/儿化/功能词链（含结构槽位）
NEUTRAL_T = [
    "为您{v}了", "请您{v}一下", "帮他{v}一下子", "我把情况说给他听",
    "{v}了再说也不迟", "好的，这就来处理", "拿了药再走也不迟", "跟他说了{q}遍啦",
    "喝了热水会好些", "您的意思我明白啦", "这事儿交给我吧", "看着点儿输液瓶",
    "把被子盖好点儿", "有什么需要尽管说", "咱们慢慢来不着急", "麻烦您签个字",
    "等一下儿就叫到您", "给我说说哪儿不舒服", "这块儿疼不疼呀", "吃点儿东西再走",
    "您先坐会儿，我这就去叫", "这事儿得跟家里商量商量", "门儿开着呢您请进",
    "{b}的家属呢，请您过来{q}趟", "这事儿还得问问大夫", "把药拿好嘞别掉了",
    "您哪儿不舒服呀，慢慢说", "给他{v}完{q}趟了再来", "麻烦您在这儿按个手印",
    "这张单子填好了交给护士", "您扶着点儿他，慢点走", "叫{nm}的家属接一下电话",
    "把化验单拿去给大夫看看", "这药吃着呢，别停", "您歇着吧，剩下的我们来弄",
    "回头把片子取一下啊", "床头卡给我瞅瞅", "输完这瓶就差不多了",
    "让{nm}过来搭把手", "记着点儿吃药，别忘啦",
]
NEUTRAL_V = ["讲", "说", "看", "拿", "送", "填", "签", "按", "扶", "递", "跑", "喊"]
# 易丢音节：动宾紧凑/双名+数量（槽位：床/名/时/量）
DROP_T = [
    "系统检测到异常将自动上报值班护士",
    "请携带就诊卡到{f}楼收费处缴费",
    "您的检查安排在{dd}{tm}{h}点{mi}",
    "{b}患者{nm}呼叫护士站请尽快前往处理",
    "{b}患者{nm}的家属请到{d}领取药品",
    "重症监护室{b}呼叫",
    "请喝{ml}毫升温水",
    "{b}患者{nm}请到{d}",
    "{b}呼叫护士",
    "呼叫转移成功请稍候",
    "通知{nm}{t}，{d}设备故障已报修",
    "{b}的{proc}安排在{dd}{tm}{h}点{mi}",
]


def han(n):
    """整数 -> 念法（1..999）。"""
    if n < 10:
        return DIGITS[n]
    if n < 20:
        return "十" + (DIGITS[n % 10] if n % 10 else "")
    if n < 100:
        return DIGITS[n // 10] + "十" + (DIGITS[n % 10] if n % 10 else "")
    s = DIGITS[n // 100] + "百"
    r = n % 100
    if r == 0:
        return s
    if r < 10:
        return s + "零" + DIGITS[r]
    return s + han(r)


def phone(digits):
    return "".join(YAO if d == "1" else DIGITS[int(d)] for d in digits)


def temp_str(v):
    i, f = int(v), int(round((v - int(v)) * 10))
    return han(i) + "点" + DIGITS[f]


def time_slot(rnd):
    tm = rnd.choice(TIMES_M)
    hour = {"凌晨": (0, 5), "清晨": (5, 7), "上午": (7, 11), "中午": (11, 13),
            "下午": (13, 18), "晚上": (18, 23)}[tm]
    h = rnd.randint(*hour)
    hs = "零" if h == 0 else (han(h) if h <= 12 else han(h - 12))
    return tm, hs


def slots(rnd):
    tm, h = time_slot(rnd)
    return {
        "b": han(rnd.randint(1, 99)) + "床" if rnd.random() < 0.75
             else "零" + DIGITS[rnd.randint(1, 9)] + "床",
        "nm": rnd.choice(SURNAMES) + rnd.choice(GIVEN),
        "t": rnd.choice(["主任", "护士长", "医生", "老师"]),
        "d": rnd.choice(DEPTS),
        "f": rnd.choice(DIGITS[1:10]),
        "tm": tm,
        "dd": rnd.choice(["今天", "明天", "后天", "周" + rnd.choice("一二三四五六日"), "星期天"]),
        "h": h,
        "mi": rnd.choice(MINUTES),
        "ml": han(rnd.choice([50, 100, 150, 200, 250, 300])),
        "mg": han(rnd.choice([5, 10, 20, 25, 50, 100, 250])),
        "q": DIGITS[rnd.randint(2, 5)],
        "v": rnd.choice(NEUTRAL_V),
        "proc": rnd.choice(PROC),
    }


def fill(t, s, rnd):
    out = t
    for k, v in s.items():
        out = out.replace("{" + k + "}", v)
    return out.replace("{d}", rnd.choice(DEPTS)).replace("{q}", rnd.choice("三四五"))


def gen(n, rnd):
    quota = {
        "小数体温": .12, "床号播报": .18, "字母科室": .08, "数字串": .08,
        "时间预约": .10, "药量操作": .14, "轻声功能词": .10, "易丢音节": .10,
        "姓名长句": .10, "安全提醒": .02,
    }
    rows, seen = [], set()

    def add(text, cat):
        if text not in seen:
            seen.add(text)
            rows.append((text, cat))
            return True
        return False

    def loop(cat, frac, make, tries=3000):
        want = int(n * frac)
        got = tries
        while sum(1 for _, c in rows if c == cat) < want and tries > 0:
            add(make(rnd), cat)
            tries -= 1
        if sum(1 for _, c in rows if c == cat) < want:
            print(f"[warn] {cat} 只有配额的不足（模板池耗尽）")

    def m_temp(rnd):
        s = slots(rnd)
        v = round(rnd.uniform(35.0, 42.0), 1)
        if abs(v - int(v)) < 1e-9:
            v = round(min(v + 0.5, 42.0), 1)
        t = temp_str(v)
        return fill(rnd.choice([
            "体温{tp}度", "患者{b}体温{tp}度", "复测体温{tp}度请记录",
            "{nm}的体温是{tp}度", "当前体温{tp}度属正常范围",
            "体温已达{tp}度请立即处理", "{nm}{b}今晨体温{tp}度已记录",
        ]).replace("{tp}", t), s, rnd)

    def m_bed(rnd):
        s = slots(rnd)
        return fill("{b}" + rnd.choice(ACTIONS), s, rnd)

    def m_letter(rnd):
        s = slots(rnd)
        l = rnd.choice(LETTERS)
        o = fill(rnd.choice(LETTER_OBJ), s, rnd)
        a = fill(rnd.choice(ACTIONS), s, rnd)
        return f"{l} {o}{a}"

    def m_num(rnd):
        s = slots(rnd)
        r = rnd.random()
        if r < 0.4:
            return fill("请拨打", s, rnd) + phone("".join(rnd.choice("0123456789") for _ in range(rnd.randint(5, 11))))
        if r < 0.6:
            return fill("床号{n}请确认", s, rnd).replace("{n}", phone(str(rnd.randint(1, 999)).zfill(3)))
        if r < 0.8:
            return fill("输液还剩{ml}毫升", s, rnd)
        return fill("剂量{mg}毫克每日{q}次", s, rnd)

    def m_time(rnd):
        s = slots(rnd)
        return fill(rnd.choice([
            "现在是{dd}{tm}{h}点{mi}", "检查安排在{dd}{tm}{h}点{mi}",
            "请于{tm}{h}点{mi}前空腹", "用药时间是{tm}{h}点{mi}",
            "{nm}的{proc}安排在{dd}{tm}{h}点{mi}",
            "手术定于{dd}{tm}{h}点{mi}请做好术前准备",
        ]), s, rnd)

    def m_drug(rnd):
        s = slots(rnd)
        return fill(rnd.choice([
            "{b}患者{nm}请做{proc}", "{proc}已完成请休息",
            "请喝{ml}毫升温水", "药已发放请服用{q}片",
            "{proc}用药{mg}毫克请核对", "一针{rnd_a}药即将注射",
            "{proc}需要注意不良反应", "请按时服用{rnd_y}每日{q}次",
            "{nm}的{proc}已停止请知悉", "注射胰岛素后请于{q}十分钟内进食",
            "{rnd_y}{q}片{rnd_w}口服", "{rnd_y}{mg}毫克{rnd_w}{rnd_f}",
            "{b}该换药了，请{rnd_y}{q}粒睡前服用",
            "雾化液已配好，请{rnd_w}做{proc}",
            "{rnd_y}外用药，请勿口服，{rnd_w}使用{q}次",
            "舌下含服{rnd_y}一片，五分钟后未缓解再含服一片",
        ]).replace("{rnd_a}", rnd.choice(["止痛", "退烧", "消炎", "平喘"]))
         .replace("{rnd_y}", rnd.choice(["降压药", "降糖药", "抗生素", "退烧药", "止痛药",
                                          "胃黏膜保护剂", "他莫昔芬", "布洛芬", "阿莫西林",
                                          "硝苯地平", "二甲双胍", "阿司匹林", "氯吡格雷"]))
         .replace("{rnd_w}", rnd.choice(["饭后", "饭前", "睡前", "早晚各一次", "空腹"]))
         .replace("{rnd_f}", rnd.choice(["一日三次", "隔日一次", "每周一次", "必要时服用"])), s, rnd)

    def m_neutral(rnd):
        s = slots(rnd)
        return fill(rnd.choice(NEUTRAL_T), s, rnd)

    def m_drop(rnd):
        s = slots(rnd)
        return fill(rnd.choice(DROP_T), s, rnd)

    def m_name(rnd):
        s = slots(rnd)
        return fill(rnd.choice([
            "{b}患者{nm}请到{d}进行{proc}，请携带就诊卡",
            "{nm}您好，您的{proc}报告已出，请到{d}领取",
            "请{nm}家属到{d}谈话，其他患者请勿围观",
            "{b}{nm}输液反应已处理，值班医生将继续观察{q}十分钟",
            "{nm}{t}请到{d}会诊",
            "{b}患者{nm}明日手术，请{t}完成术前谈话",
        ]), s, rnd)

    def m_remind(rnd):
        return rnd.choice(REMIND) + "，" + rnd.choice(REMIND)

    makers = {
        "小数体温": m_temp, "床号播报": m_bed, "字母科室": m_letter, "数字串": m_num,
        "时间预约": m_time, "药量操作": m_drug, "轻声功能词": m_neutral,
        "易丢音节": m_drop, "姓名长句": m_name, "安全提醒": m_remind,
    }
    for cat, frac in quota.items():
        loop(cat, frac, makers[cat])

    out = []
    seen2 = set()
    for text, cat in rows:
        if text not in seen2:
            seen2.add(text)
            out.append((text, cat))
    rnd.shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--out", default="data/record_script_ext.tsv")
    ap.add_argument("--start-id", type=int, default=101)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    rows = gen(args.n, rnd)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# 播报扩展录制稿 {len(rows)} 条（make_record_script.py seed={args.seed} 生成）\n")
        f.write("# 列: id <TAB> 录制文本 <TAB> 类别；数字全汉字、字母逐个念\n")
        for i, (text, cat) in enumerate(rows, args.start_id):
            f.write(f"u{i:04d}\t{text}\t{cat}\n")
    cnt = collections.Counter(c for _, c in rows)
    print(f"wrote {args.out}: {len(rows)} 条")
    for c, k in cnt.most_common():
        print(f"  {c:8s} {k}")


if __name__ == "__main__":
    main()
