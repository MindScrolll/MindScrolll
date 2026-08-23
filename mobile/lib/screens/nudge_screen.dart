import 'package:flutter/material.dart';

class NudgeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.blueGrey[50],
      appBar: AppBar(
        title: const Text('Zihinsel Mola Alanı', style: TextStyle(color: Colors.black87)),
        backgroundColor: Colors.white,
        elevation: 0,
        iconTheme: const IconThemeData(color: Colors.black87),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16.0),
        children: [
          _buildNudgeCard(
            title: 'Doğaya Dönüş: Dijital Mola',
            description: 'Ekrandan uzaklaşıp temiz hava almak, stres seviyenizi düşürerek zihinsel yorgunluğu azaltır.',
            icon: Icons.nature_people,
            iconColor: Colors.green,
          ),
          const SizedBox(height: 12),
          _buildNudgeCard(
            title: 'Bilinçli bir mola zamanı',
            description: 'Kesintisiz uyaran akışı odaklanma süresini kısaltır. Birkaç dakika sadece ana odaklanarak zihninizi tazeleyin.',
            icon: Icons.coffee,
            iconColor: Colors.brown,
          ),
        ],
      ),
    );
  }

  Widget _buildNudgeCard({required String title, required String description, required IconData icon, required Color iconColor}) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Row(
          children: [
            Icon(icon, size: 40, color: iconColor),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  Text(description, style: const TextStyle(fontSize: 14, color: Colors.black54)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}