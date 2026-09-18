//! 纠错服务模式：神经模型常驻后台，stdin 一行一个拼音、stdout 一行一个纠正结果。
//!
//! 协议：一行一个连写拼音，回一行纠正结果（模型没提名就回空行）；空行跳过；EOF 退出。
//! 日志走 stderr，stdout 只留结果，调用方按行对齐读取。
//!
//! 后台跑（release 构建，首个查询会触发 Metal/CPU 预热）：
//! ```bash
//! cargo build --release -p qingjian-cli
//! nohup ./target/release/qingjian-cli --neural-corrector data/corrector --corrector-serve \
//!   >/tmp/corrector-service.log 2>&1 &
//! ```
//! Python 侧用 subprocess 管道驱动（训练循环里反复问，不用每次加载 28MB 权重）：
//! ```python
//! import subprocess
//! proc = subprocess.Popen(
//!     ["./target/release/qingjian-cli", "--neural-corrector", "data/corrector",
//!      "--corrector-serve"],
//!     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
//! )
//! proc.stdin.write("zhoongguo\n")
//! proc.stdin.flush()
//! print(proc.stdout.readline(), end="")  # zhongguo
//! ```

use std::io::{self, BufRead, Write};

use qingjian_core::correction::PinyinCorrector;

use crate::args::Args;
use crate::error::CliError;

/// 只加载神经纠错模型（不加载词库与语言模型），逐行读 stdin、逐行写纠正结果。
pub fn run(args: &Args) -> Result<(), CliError> {
    let Some(dir) = args.neural_corrector.as_ref() else {
        return Err(CliError::MissingCorrector);
    };
    let corrector = qingjian_neural::corrector::NeuralCorrector::load(dir)?;
    eprintln!(
        "神经纠错服务就绪（{}），一行一个拼音，EOF 退出。",
        dir.display()
    );
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let input = line?;
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        let fixed = corrector
            .correct(input)
            .into_iter()
            .next()
            .unwrap_or_default();
        writeln!(out, "{fixed}")?;
        out.flush()?;
    }
    Ok(())
}
