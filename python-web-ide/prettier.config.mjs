/**
 * 项目统一格式化配置。
 * 该文件使用 JavaScript 模块格式而非 JSON，因此可保留中文说明，且会被 Prettier 自动识别。
 */
export default {
  // 与现有 TypeScript 代码风格保持一致，不在每条语句末尾强制分号。
  semi: false,
  // TypeScript 和 TSX 字符串优先使用单引号，JSX 属性值仍由 Prettier 按语法处理。
  singleQuote: true,
  // 使用两个空格缩进，降低初学者阅读嵌套结构的负担。
  tabWidth: 2,
  // 需要多行时保留末尾逗号，便于后续增删数组项或对象属性。
  trailingComma: 'all',
  // 控制单行长度，配合 VS Code 的 100 列参考线保持可读性。
  printWidth: 100,
}
