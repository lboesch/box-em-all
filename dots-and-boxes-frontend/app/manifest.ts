import { MetadataRoute } from 'next'

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: 'Dots and Boxes',
    short_name: 'Dots & Boxes',
    description: 'A fun Dots and Boxes game',
    start_url: '/',
    display: 'standalone',
    background_color: '#f3f4f6',
    theme_color: '#3b82f6',
    icons: [
      {
        src: '/icon-192x192.png',
        sizes: '192x192',
        type: 'image/png',
      },
      {
        src: '/icon-512x512.png',
        sizes: '512x512',
        type: 'image/png',
      },
    ],
  }
}

