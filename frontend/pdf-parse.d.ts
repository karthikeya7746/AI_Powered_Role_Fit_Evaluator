declare module 'pdf-parse' {
  function pdfParse(buffer: Buffer, options?: unknown): Promise<{ text?: string }>;
  export default pdfParse;
}
