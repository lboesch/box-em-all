import "@/styles/main.css";

export const metadata = {
  title: "Dot and Boxes",
  description: "A fun Dots and Boxes game",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
