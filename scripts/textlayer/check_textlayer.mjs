#!/usr/bin/env node
// 文本层书面边界改写的回归台：**直接跑调用方仓库里的真源码**，不复制一份进来。
//
// 为什么要它：本仓的主线结论是"不改模型、只改喂进去的书面边界就能降错率"
// （24.2%→3.6%，见 docs/icefall-finetune-flow.md），而落地实现在另一个仓的 ArkTS 文件里
// （RtcTtsText.ets 的 normalizeTtsText / splitLongSentence / markQuantityBoundary）。
// 两边各写一份期望就会漂移，所以这里把**真源码**和**对方的 Hypium 测试文件**一起读进来跑。
//
// 为什么不用现成的 TS 运行器：开发机上的 node 是 18，不支持 `.ts` 类型剥离（Node 22.6+ 才可行），
// 而 ArkTS 这里用到的类型注解很规整，正则去掉即可；去掉后若还有语法错误会打印行号，不会静默跑偏。
//
// 用法：
//   RTC_SRC=/path/to/src/main/ets/tts/RtcTtsText.ets node scripts/textlayer/check_textlayer.mjs
//   RTC_TEST=/path/to/test/RtcTtsText.test.ets node scripts/textlayer/check_textlayer.mjs   # 再跑一遍对方断言
// 两个变量都不给就只打印用法（不猜对方仓库在哪）。
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const SRC = process.env.RTC_SRC || "";
const TEST = process.env.RTC_TEST || "";
if (!SRC) {
  console.log("用法：RTC_SRC=<RtcTtsText.ets 路径> [RTC_TEST=<RtcTtsText.test.ets 路径>] node 本脚本");
  process.exit(2);
}

// --- 1) 去掉类型注解（只针对 string/number/boolean 这一小组，够用且不碰字符串字面量）---
function strip(src) {
  return src
    .replace(/: ?(string|number|boolean)(\[\])?([,)=])/g, "$3")
    .replace(/: ?(string|number|boolean)(\[\])?(\s*(=>|\{))/g, "$2$3")
    .replace(/: ?(string|number|boolean)\[\] =/g, " =");
}

const outDir = fs.mkdtempSync(path.join(os.tmpdir(), "textlayer-"));
const mjs = path.join(outDir, "rtc_tts_text.mjs");
fs.writeFileSync(mjs, strip(fs.readFileSync(SRC, "utf8")), "utf8");

let mod;
try {
  mod = await import(pathToFileURL(mjs).href);
} catch (e) {
  console.log(`!! 剥离类型后仍无法加载：${e.message}`);
  console.log("   多半是出现了这里没覆盖的 TS 写法，把那条注解补进 strip() 而不是复制源码。");
  process.exit(1);
}
const { normalizeTtsText } = mod;

// --- 2) 期望表：每组都注明它锁的是哪次实测 ---
// 组一：数量短语后加句号（PC 各 160 次回读 24.2%→3.6%；真机四档 A/B 见 docs/device-runtime-check.md）
const QUANTITY = [
  ["每次六点七毫克，服用后请漱口", "每次六点七毫克，服用后请漱口"], // 已带标点 → 整句放行
  ["六十九床请调整体位", "六十九床。请调整体位"],
  ["一百三十八床呼叫护士站", "一百三十八床。呼叫护士站"],
  ["三十三床需要换纱布", "三十三床。需要换纱布"],
  ["中午二点五十分心电监护", "中午二点五十分。心电监护"],
  ["清晨七点五十分雾化", "清晨七点五十分。雾化"],
  ["每次六点七毫克服用后请漱口", "每次六点七毫克。服用后请漱口"],
  ["体温36.5度呼叫护士站", "体温36.5度。呼叫护士站"],   // 现网真实输入是阿拉伯数字
  ["21床正在呼叫", "21床。正在呼叫"],                 // 现场 0% 错率也照加（用户拍的"所有都加"）
  ["10:36复查", "十点三十六分。复查"],
];
// 组二：绝不能造的假阳性
const GUARDS = [
  ["十分感谢配合治疗", "十分感谢配合治疗"],            // "十分"是副词；切了会出"十。分感谢"
  ["三床请到护士站", "三床请到护士站"],                // 单字数量不加
  ["体温36.5度", "体温36.5度"],                       // 数量在句尾，后面没内容
  ["30分钟后复查", "30分钟后复查"],                   // 数量+方位词是一整个短语
  ["37.5度以下物理降温", "37.5度以下物理降温"],
  ["住院号零一一七七一二五", "住院号零一一七七一二五"],  // 长编号已逐位读，无单位
  ["比例25:99", "比例二十五比九十九"],                 // 非时刻的冒号按比分读，不再留裸冒号
];
// 组三：更早定档、不许被这次改动破坏的行为
const REGRESS = [
  ["10:36，请3床患者到护士站", "十点三十六分，请3床患者到护士站"],
  ["08:05 复查", "八点零五分 复查"],
  ["住院号01177125", "住院号零一一七七一二五"],
  ["电话0755-1234567", "电话零七五五-一二三四五六七"],  // 连字符不能念成"减"
  ["3+2=5，9-4=5，6×3=18，8÷2=4", "3加2等于5，9减4等于5，6乘3等于18，8除2等于4"],
  ["ABCDEFGHIJ", "A;B;C;D;E;F;G;H;I;J;"],             // 字母逐个、用分号
  ["PICC维护", "P;I;C;C;维护"],
  ["Nurse station", "Nurse station"],                 // 含小写的正常英文整词读
  ["二十三床患者张三呼叫护士站请尽快前往处理", "二十三床患者张三,呼叫护士站,请尽快前往处理"], // 长句两刀
  ["本楼层设备巡检已完成状态正常无需处理", "本楼层设备巡检已完成状态正常无需处理"],         // 无锚点不硬切
  ["", ""],
];

let bad = 0;
function run(label, cases) {
  console.log(`\n== ${label} ==`);
  for (const [src, want] of cases) {
    const got = normalizeTtsText(src);
    const ok = got === want;
    if (!ok) bad++;
    console.log(`${ok ? "OK  " : "FAIL"} in=${src}\n     out=${got}\n     want=${want}`);
  }
}
run("数量短语后加句号", QUANTITY);
run("不能造的假阳性", GUARDS);
run("已定档行为不许破", REGRESS);

// --- 3) 把对方 Hypium 里的断言也抠出来跑，防两边期望漂移 ---
if (TEST && fs.existsSync(TEST)) {
  console.log("\n== 对方仓库 Hypium 断言 ==");
  const raw = fs.readFileSync(TEST, "utf8").replace(/\r?\n\s*\.assertEqual/g, ".assertEqual");
  const IN = "normalizeTtsText('";
  const MID = "')).assertEqual('";
  const END = "')";
  let from = 0, n = 0, tbad = 0;
  for (;;) {
    const a = raw.indexOf(IN, from);
    if (a < 0) break;
    const b = raw.indexOf(MID, a);
    if (b < 0) break;
    const c = raw.indexOf(END, b + MID.length);
    if (c < 0) break;
    const src = raw.slice(a + IN.length, b);
    const want = raw.slice(b + MID.length, c);
    const got = normalizeTtsText(src);
    n++;
    if (got !== want) {
      tbad++;
      bad++;
      console.log(`FAIL in=${src}\n     out=${got}\n     want=${want}`);
    }
    from = c + END.length;
  }
  console.log(`hypothesis_asserts=${n} fails=${tbad}`);
}

fs.rmSync(outDir, { recursive: true, force: true });
console.log(bad === 0 ? "\nALL_AS_EXPECTED" : `\nMISMATCH_COUNT=${bad}`);
process.exit(bad === 0 ? 0 : 1);
