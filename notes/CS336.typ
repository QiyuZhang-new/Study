// 封面变量，统一管理封面内容
#let cover = (
  title: "CS336笔记",
  subtitle: "",
  author_line: [#link("https://zhangqiyu.me")[Qiyu Zhang] 编著],
  logo: "myimage.jpg", // 相对或绝对路径
  publisher: [#link("https://zhangqiyu.me")[https://zhangqiyu.me]], 
  date: "2025年11月",
)


//设置全局字体样式与大小
#set text(
  font:("Consolas","Simsun"),
  size:12pt,
  // lang:"zh",
  // region:"cn"
)

//设置首行缩进与行距
#set par(
  first-line-indent: (amount: 0em, all: false),
  justify: true,//两端对齐
  leading: 0.8em,
)

//#show selector: it => { ... }：为匹配到的节点定义渲染器，it 是原始的标题内容（保留它可使内容按原样插入）

#show heading.where(level:1): it => {
  v(0em)
  set align(center)//居中
  set text(
    font:("STKaiti"),
    size:20pt,
    weight:"bold",//默认加粗
  )
  it
  v(2em)
}

#show heading.where(level:2): it => {
  v(0em)
  set align(left)//居中
  set text(
    font:("SimHei"),
    size:18pt,
    weight:"bold",//默认加粗
  )
  it
  v(1em)
}

#show heading.where(level:3): it => {
  v(2em)
  set align(center)
  set text(
    font:("SimHei"),
    size:16pt,
    weight:"bold",//默认加粗
  )
  it
  v(1em)
}

#show heading.where(level:4): it => {
  v(0em)
  set align(left)//居中
  set text(
    font:("SimHei"),
    size:15pt,
    weight:"bold",//默认加粗
  )
  it
  v(2em)
}


#show link: it => {
  set text(blue)
  it
}
// ==================== 封面页 ====================
#page(
  margin: 0cm,
  header: none,
  footer: none,
  numbering: none,
)[
  #align(center)[
    #v(3cm)
    #text(size: 36pt, weight: "bold")[#cover.title]
    #v(1cm)
    #text(size: 20pt)[#cover.subtitle]
    #v(4cm)
    #text(size: 16pt, font:"Consolas")[#cover.author_line]
    #v(5cm)
    #image("../myimage.jpg", width: 5cm)  // 图片参数为表达式，保持不变
    #v(1cm)
    #text(size: 14pt, font:"Consolas")[#cover.publisher]
    #v(0.5cm)
    #text(size: 14pt, font:"STKaiti")[#cover.date]
  ]
]


#set page(numbering: none)
#outline(
  depth:none,//显示目录的深度
  indent:2em,//目录缩进
  target:heading.where(outlined:true),//目录中应包含的元素类型，默认为heading.where(outlined:true)
  title:[目录],//默认为一级标题
)


#set page(
  paper:"a4",
  margin:(top: 2cm, bottom: 2cm, left: 2cm, right: 2cm),
  numbering: "1",
  number-align: center,

  footer: context {
    align(center)[
      #v(0em)
      #counter(page).display()
    ]
  }
)


#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)//重置页码

#show strong: set text(purple, weight: "bold")
//#v(3em)

#let at(con) = {
  text(font: "STKaiti", size: 12pt, weight: "bold")[#con]
  v(1em)
}

#let the(con)={
  text(red)[#con]
}

#import "@preview/fletcher:0.5.8": diagram, node, edge